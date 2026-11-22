Paste this into Claude Code (run `claude` inside ~/Projects/cancer-fingerprints):

---

You are the project lead for Cancer Fingerprints. Read CLAUDE.md. You have a team of subagents in .claude/agents/: data-engineer, bioinformatician, software-engineer and reviewer. Delegate work to them rather than doing it all yourself, and run independent pieces in parallel.

Do Phase 1 (CLAUDE.md Milestones 1-2) now:
- software-engineer: package scaffold with uv, ruff, pytest, GitHub Actions CI, CLI entry point
- data-engineer: download_data.py and download_signatures.py
- bioinformatician: profile.py (MAF -> 96-channel SBS matrix) with exact-count tests on a tiny fixture

After each piece, have the reviewer check it, fix what it finds, then commit on a branch `phase-1` with a clear message. When Phase 1 is done, push the branch, open a pull request, and give me a short summary plus any biology questions for me. Do not start Phase 2 until I approve.
