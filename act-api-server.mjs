/*
  心向本机 AI 服务。
  启动：$env:DEEPSEEK_API_KEY='你的密钥'; node .\act-api-server.mjs
  访问：http://127.0.0.1:8788/act-app.html
  仅监听本机；请求内容不会写入磁盘或日志。
*/
import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const root = dirname(fileURLToPath(import.meta.url));
const PORT = Number(process.env.PORT || process.env.ACT_PORT || 8788);
const APP_REVISION = 'online-diagnostics-v1';
const PROVIDER = process.env.AI_PROVIDER || 'deepseek';
const DEEPSEEK_MODEL = process.env.DEEPSEEK_MODEL || 'deepseek-v4-flash';
const OPENAI_MODEL = process.env.OPENAI_MODEL || 'gpt-5.6-terra';
const REQUEST_LIMIT = 10_000;
const MODEL_TIMEOUT_MS = 15_000;
const STAGES = new Set(['value', 'action']);
const SKILLS = new Set(['present', 'accept', 'defusion', 'self', 'values', 'commit', '']);

const schema = {
  type: 'object', additionalProperties: false, required: ['message', 'suggested_action', 'value_direction'],
  properties: {
    message: { type: 'string', maxLength: 280 },
    suggested_action: { type: 'string', maxLength: 160 },
    value_direction: { type: 'string', maxLength: 160 }
  }
};

function json(res, status, body) {
  res.writeHead(status, { 'content-type': 'application/json; charset=utf-8', 'cache-control': 'no-store', 'x-content-type-options': 'nosniff' });
  res.end(JSON.stringify(body));
}
function cleanText(value, max) {
  return typeof value === 'string' ? value.replace(/[\u0000-\u001f]/g, ' ').trim().slice(0, max) : '';
}
function contextFrom(value) {
  if (!value || typeof value !== 'object' || Array.isArray(value) || !STAGES.has(value.stage)) return null;
  const context = {
    stage: value.stage,
    domain: cleanText(value.domain, 48),
    value: cleanText(value.value, 240),
    obstacle: cleanText(value.obstacle, 240),
    action: cleanText(value.action, 160),
    timing: cleanText(value.timing, 160),
    minimum_acceptable: cleanText(value.minimum_acceptable, 160),
    skill: SKILLS.has(value.skill) ? value.skill : ''
  };
  if (context.stage === 'value' && !context.value) return null;
  if (context.stage === 'action' && (!context.value || !context.action)) return null;
  return context;
}
function safeText(value, max) {
  if (typeof value !== 'string') return '';
  const text = value.replace(/[\u0000-\u001f]/g, ' ').trim().slice(0, max);
  if (!text || /(诊断|疾病|治疗|治愈|药物|剂量|处方|自杀风险评估|保证)/.test(text)) return '';
  return text;
}
function inputAnchors(text) {
  const normalized = cleanText(text, 240).replace(/[，。！？、；：\s]/g, '');
  const anchors = new Set();
  for (let index = 0; index < normalized.length - 1; index += 1) anchors.add(normalized.slice(index, index + 2));
  return [...anchors].filter(anchor => !/^(一个|一下|然后|之后|今天|自己|可以|需要)$/.test(anchor));
}
function refersToUserInput(text, context) {
  const source = context.stage === 'action' ? `${context.action} ${context.timing} ${context.minimum_acceptable} ${context.value}` : `${context.value}`;
  const anchors = inputAnchors(source);
  return anchors.length === 0 || anchors.some(anchor => text.includes(anchor));
}
function safeResponse(payload, context) {
  const message = safeText(payload?.message, 280);
  const suggestedAction = context.stage === 'action' ? safeText(payload?.suggested_action, 160) : '';
  const valueDirection = context.stage === 'value' ? safeText(payload?.value_direction, 160) : '';
  if (!message || !refersToUserInput(`${message} ${suggestedAction} ${valueDirection}`, context)) return null;
  return { message, suggested_action: suggestedAction, value_direction: valueDirection };
}
function publicError(error) {
  const message = error instanceof Error ? error.message : '';
  if (message.includes('DEEPSEEK_API_KEY') || message.includes('OPENAI_API_KEY')) return '服务端尚未配置 API Key。';
  if (message.includes('Unsupported AI_PROVIDER')) return 'AI_PROVIDER 配置无效。';
  if (message.includes('abort')) return 'AI 服务响应超时。';
  if (message.includes('401')) return 'API Key 无效或已失效。';
  if (message.includes('402')) return 'DeepSeek 账户余额不足或当前额度不可用。';
  if (message.includes('403')) return 'DeepSeek 拒绝了此请求，请检查 Key 的权限或账号状态。';
  if (message.includes('400')) return 'DeepSeek 拒绝了请求参数，请检查模型配置。';
  if (message.includes('429')) return 'AI 服务暂时限流，请稍后重试。';
  if (message.includes('Model returned invalid content')) return 'AI 返回内容未通过格式校验，请再试一次。';
  if (message.includes('fetch failed')) return '服务端暂时无法连接 DeepSeek。';
  return 'AI 服务暂时不可用，请稍后重试。';
}
async function fetchWithTimeout(url, options) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), MODEL_TIMEOUT_MS);
  try { return await fetch(url, { ...options, signal: controller.signal }); }
  finally { clearTimeout(timer); }
}
function instructions(context) {
  const purpose = context.stage === 'value'
    ? 'Help distinguish a result wish from a value direction. Explicitly reference the user’s words, then offer one optional value_direction phrased as an ongoing way of treating oneself or others, not as an outcome, feeling, achievement, or productivity target. Ask one short clarification question in message. suggested_action must be empty.'
    : 'Respond with a grounded, everyday reflection and one optional smaller version of the user’s action. You MUST explicitly reference at least one concrete detail from the stated action or value. Do not introduce unrelated activities, body-care advice, or new goals. Tie it to the stated value and, where helpful, offer a practical first sentence, boundary, or condition. The smaller action must be concrete, doable today, take ten minutes or less, and remain entirely optional.';
  return `You are a supportive Chinese ACT self-practice writing assistant for a personal local app. ${purpose}

Hard boundaries: This is not therapy, diagnosis, treatment, crisis support, or medical advice. Do not assess safety, mention disorders, claim outcomes, prescribe medication, or make promises. Do not shame, command, or imply the user must eliminate thoughts or feelings. Use natural, plain, non-judgmental Chinese, like a thoughtful companion rather than a textbook. Do not name ACT concepts unless the user asks. Do not merely repeat the user’s wording, but do use a short concrete phrase from it so the response is unmistakably about this user’s situation. Values are directions, not achievement targets. Keep the message under 120 Chinese characters. Return JSON only with message, suggested_action and value_direction. Use empty strings for fields not applicable to the stage.

The user submitted this current exercise context: ${JSON.stringify(context)}`;
}
async function callDeepSeek(context) {
  if (!process.env.DEEPSEEK_API_KEY) throw new Error('DEEPSEEK_API_KEY is not configured');
  const response = await fetchWithTimeout('https://api.deepseek.com/chat/completions', {
    method: 'POST', headers: { authorization: `Bearer ${process.env.DEEPSEEK_API_KEY}`, 'content-type': 'application/json' },
    body: JSON.stringify({ model: DEEPSEEK_MODEL, messages: [{ role: 'system', content: instructions(context) }], response_format: { type: 'json_object' }, thinking: { type: 'disabled' }, max_tokens: 400, stream: false })
  });
  if (!response.ok) throw new Error(`DeepSeek request failed: ${response.status}`);
  const data = await response.json();
  return JSON.parse(data.choices?.[0]?.message?.content || '');
}
async function callOpenAI(context) {
  if (!process.env.OPENAI_API_KEY) throw new Error('OPENAI_API_KEY is not configured');
  const response = await fetchWithTimeout('https://api.openai.com/v1/responses', {
    method: 'POST', headers: { authorization: `Bearer ${process.env.OPENAI_API_KEY}`, 'content-type': 'application/json' },
    body: JSON.stringify({ model: OPENAI_MODEL, instructions: instructions(context), input: 'Return the requested JSON object.', text: { format: { type: 'json_schema', name: 'act_guidance', strict: true, schema } } })
  });
  if (!response.ok) throw new Error(`OpenAI request failed: ${response.status}`);
  const data = await response.json();
  return JSON.parse(data.output_text || '');
}
async function createResponse(context) {
  let raw;
  if (PROVIDER === 'deepseek') raw = await callDeepSeek(context);
  else if (PROVIDER === 'openai') raw = await callOpenAI(context);
  else throw new Error('Unsupported AI_PROVIDER');
  const result = safeResponse(raw, context);
  if (!result) throw new Error('Model returned invalid content');
  return result;
}

createServer(async (req, res) => {
  const url = new URL(req.url, `http://${req.headers.host}`);
  const origin = req.headers.origin;
  if (origin === 'null' || origin === 'http://127.0.0.1:8788' || origin === `http://localhost:${PORT}` || origin === `http://127.0.0.1:${PORT}`) res.setHeader('access-control-allow-origin', origin);
  if (req.method === 'OPTIONS' && url.pathname.startsWith('/api/')) {
    res.writeHead(204, { 'access-control-allow-methods': 'POST, OPTIONS', 'access-control-allow-headers': 'content-type', 'access-control-max-age': '600' }); return res.end();
  }
  if (req.method === 'GET' && url.pathname === '/api/health') {
    const configured = PROVIDER === 'deepseek' ? Boolean(process.env.DEEPSEEK_API_KEY) : PROVIDER === 'openai' ? Boolean(process.env.OPENAI_API_KEY) : false;
    return json(res, 200, { ok: configured, provider: PROVIDER, model: PROVIDER === 'deepseek' ? DEEPSEEK_MODEL : OPENAI_MODEL, revision: APP_REVISION, message: configured ? 'AI 服务已配置。' : '服务端尚未配置 API Key。' });
  }
  if (req.method === 'POST' && (url.pathname === '/api/act/guide' || url.pathname === '/api/act/reflection')) {
    if (!String(req.headers['content-type'] || '').toLowerCase().startsWith('application/json')) return json(res, 415, { error: 'Expected JSON request' });
    let raw = '';
    for await (const chunk of req) { raw += chunk; if (raw.length > REQUEST_LIMIT) return json(res, 413, { error: 'Request too large' }); }
    let context;
    try { context = contextFrom(JSON.parse(raw || '{}')); } catch { return json(res, 400, { error: 'Invalid JSON request' }); }
    if (!context) return json(res, 400, { error: 'Invalid ACT context' });
    if (url.pathname.endsWith('/guide') && context.stage !== 'value') return json(res, 400, { error: 'Invalid guide stage' });
    if (url.pathname.endsWith('/reflection') && context.stage !== 'action') return json(res, 400, { error: 'Invalid reflection stage' });
    try { return json(res, 200, await createResponse(context)); }
    catch (error) { return json(res, 503, { error: 'AI_UNAVAILABLE', message: publicError(error) }); }
  }
  if (req.method === 'GET' && (url.pathname === '/' || url.pathname === '/act-app.html')) {
    try { const html = await readFile(join(root, 'act-app.html')); res.writeHead(200, { 'content-type': 'text/html; charset=utf-8', 'cache-control': 'no-store', 'x-content-type-options': 'nosniff', 'referrer-policy': 'no-referrer' }); return res.end(html); }
    catch { return res.end('Missing act-app.html'); }
  }
  json(res, 404, { error: 'Not found' });
}).listen(PORT, '0.0.0.0', () => console.log(`心向 ACT: http://127.0.0.1:${PORT}/act-app.html`));
