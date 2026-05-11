---
name: agent-browser
description: Browser automation for this project. Use when: opening Colab notebooks, scraping render results, clicking UI elements, taking screenshots of dashboards, or any web interaction. Runs via `agent-browser` CLI (Chrome CDP, no Playwright needed).
allowed-tools: Bash(agent-browser *)
---

# agent-browser

PATH: export PATH="$HOME/.npm-global/bin:$PATH" before every agent-browser call.

## TOKEN BUDGET RULES — follow strictly

- NEVER use `agent-browser snapshot` (dumps full accessibility tree — expensive)
- NEVER use `agent-browser get text` without a specific ref
- ALWAYS use `agent-browser screenshot` → read the image → close
- Only call `snapshot -i` when you must click something and have no URL to navigate to directly
- One screenshot per page. Do not loop or retry unless the page was blank.

## "go" workflow (user says "go" after running a Colab cell)

Colab URL: https://colab.research.google.com/drive/1RWw8Djt_JTsZyDRRzrg_EWQXEkIKgzcS?authuser=3#scrollTo=C_4n3vBWNjVo

```bash
export PATH="$HOME/.npm-global/bin:$PATH"

# 1. Colab — screenshot the last cell output
agent-browser open "https://colab.research.google.com/drive/1RWw8Djt_JTsZyDRRzrg_EWQXEkIKgzcS?authuser=3#scrollTo=C_4n3vBWNjVo"
agent-browser wait --load networkidle
agent-browser screenshot /tmp/colab_out.png
agent-browser close

# 2. If user provides a video URL, screenshot it too
# agent-browser open <video_url>
# agent-browser screenshot /tmp/video_out.png
# agent-browser close
```
Then Read /tmp/colab_out.png and report what's visible (overlay_drawn_ratio, errors, last cell output).

## Screenshot-first pattern (default for all tasks)

```bash
agent-browser open <url>
agent-browser wait --load networkidle
agent-browser screenshot /tmp/out.png
agent-browser close
```

## Only use snapshot when clicking is required

```bash
agent-browser open <url>
agent-browser wait --load networkidle
agent-browser snapshot -i           # minimal: interactive only
agent-browser click @eN
agent-browser wait --load networkidle
agent-browser screenshot /tmp/out.png
agent-browser close
```

## Install check
```bash
export PATH="$HOME/.npm-global/bin:$PATH"
agent-browser --version             # 0.27.0
```
