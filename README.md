# AI Clocktower Observer

这是一个本地单局《血染钟楼：暗流涌动》观战器。程序负责规则与说书人裁量，五名 AI 玩家自主推理、交流、私聊、提名、投票并完成游戏；人类只在一个对话式窗口中旁观正文、模型原始推理字段、工具过程、规则事件和笔记检查点。

首版刻意保持很小：只支持 Trouble Brewing、一个进程内的一局游戏、一个追加式 JSONL 历史文件。没有数据库、多局列表、长期统计、检查点恢复、导出、倍速、单步或阶段跳转。运行控制只有 Stop 和 Continue；打开的历史记录和检查点均只读。

## Windows 安装与启动

需要 Python 3.12 或更高版本，以及 Node.js 22 或更高版本。在 PowerShell 中从仓库根目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
Copy-Item config.example.toml config.toml
cd web
npm install
npm run build
cd ..
.\.venv\Scripts\python.exe -m uvicorn clocktower.main:app --host 127.0.0.1 --port 8000
```

随后打开 `http://127.0.0.1:8000`。FastAPI 同源提供前端静态文件、`/api` 控制接口与 SSE 实时事件；静态根路由不会替代 `/api`。

服务启动时会从 `config.toml` 自动开始唯一一局游戏。游戏结束后若要开始新局，请停止并重新启动进程。历史写入 `game.history_directory` 指定的目录；页面中的 Open history 通过浏览器本地文件选择器读取 JSONL，不会上传文件。

## 模型与密钥配置

首版只调用 OpenAI-compatible Chat Completions 端点。`config.toml` 中的 provider 配置只保存 API Key 对应的环境变量名；真实密钥必须放在环境变量中。例如默认配置使用：

```powershell
$env:OPENAI_API_KEY = "your-api-key"
```

不要把密钥直接写进 `config.toml`。API Key、`config.toml` 和生成的历史文件都被 Git 忽略，密钥也不会写入 JSONL 历史。

模型有四层配置：

- `models.global`：无玩家覆盖时使用的全局普通模型。
- `models.global_short`：响应意愿和结束探测等快速短调用的全局模型。
- `players.<player_id>.model`：该玩家的普通模型覆盖。
- `players.<player_id>.short_model`：该玩家的短调用模型覆盖。

普通调用按 `player.model → models.global` 回退；短调用按 `player.short_model → models.global_short → player.model → models.global` 回退。

provider 的 `reasoning_fields` 决定要保留的兼容端点原始推理字段，例如：

```toml
[providers.main]
base_url = "https://api.openai.com/v1"
api_key_env = "OPENAI_API_KEY"
reasoning_fields = ["reasoning_content", "thinking"]
```

若端点同时返回多个已配置字段，观战器按返回顺序分别保存并显示字段名和原文，不总结或改写。不同兼容端点对原始推理与工具流的支持并不相同，请按供应商实际协议调整 `base_url`、模型名与字段列表。

## 验证

运行完整后端测试：

```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

运行前端测试与生产构建：

```powershell
cd web
npm test -- --run
npm run build
cd ..
```

无网络的五人完整局与历史回放验收：

```powershell
.\.venv\Scripts\python.exe -m pytest tests/integration/test_headless_game.py tests/integration/test_history_replay.py -v
```

该 fake-provider 路径会验证固定种子完整游戏、首尾记录、连续序号、模型片段顺序和 `source_field`、笔记更新后紧邻检查点，以及玩家 prompt 的 audience 隔离。它不会访问外部网络，也不能证明任一真实兼容供应商能端到端完成一局。

当前仓库未随附真实 API Key，真实 provider 的五人完整局、供应商特有推理字段和网络流式行为需要使用者配置凭据后另行验收。
