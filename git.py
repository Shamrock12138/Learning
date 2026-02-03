#                           git相关脚本
#                           2025/12/1
#                            shamrock

import subprocess
import os, argparse, sys
from datetime import datetime
from pathlib import Path

_Green = '\033[32m'
_Red = '\033[33m'
_Blue = '\033[34m'
_End = '\033[0m'

'''
  合并操作：
    git fetch
    git switch [target]
    git merge [source]
    git push origin [target]
'''

#----------------- 要提交的文件 ----------------

submit_files = [
  '.'
]

commit_message = 'Shamrock_PC'

#----------------- 工具函数 ----------------

REPO_ROOT = Path.cwd()

def run_git_cmd(cmd, cwd=None, allow_fail=False):
  """安全执行 git 命令，返回 (stdout, stderr)"""
  try:
    result = subprocess.run(
      ['git']+cmd,
      cwd=cwd,
      capture_output=True,
      text=True,
      encoding='utf-8',
      check=not allow_fail
    )
    return result.stdout.strip(), result.stderr.strip()
  except subprocess.CalledProcessError as e:
    if allow_fail:
      return "", e.stderr.strip()
    print(_Red+f"Git错误: {' '.join(cmd)}"+_End)
    print(e.stderr)
    sys.exit(1)

def is_clean(cwd=None):
  """检查工作区是否干净"""
  stdout, _ = run_git_cmd(['status', '--porcelain'], cwd=cwd)
  return not stdout

def current_branch(cwd=None):
  branch, _ = run_git_cmd(["rev-parse", "--abbrev-ref", "HEAD"], cwd=cwd)
  return branch

#----------------- 函数 ----------------

def submit(files, message, cwd=None):
  if isinstance(files, str):
    files = [files]
  run_git_cmd(['add']+files, cwd=cwd)
  staged, _ = run_git_cmd(['diff', '--name-only', '--cached'], cwd=cwd)
  if not staged:
    print(_Green+"无变更，跳过提交"+_End)
    return
  
  if not message:
    message = f"[Auto] {datetime.now().strftime('%Y-%m-%d %H:%M')}"
  run_git_cmd(["commit", "-m", message], cwd=cwd)

  branch = current_branch(cwd)
  run_git_cmd(["push", "origin", branch], cwd=cwd)
  print(_Green+f"✓ 已提交并推送到 origin/{branch}"+_End)

def merge(source, target, cwd=None):
  if not is_clean(cwd):
    print(_Red+"工作区有未提交更改，请先 stash 或 commit"+_End)
    sys.exit(1)
  branch = current_branch(cwd)

  run_git_cmd(["switch", target], cwd=cwd)
  run_git_cmd(["fetch", "origin"], cwd=cwd)
  stdout, stderr = run_git_cmd(
    ["merge", f"origin/{source}", "--no-ff", "--no-edit"],
    cwd=cwd,
    allow_fail=True
  )
  if "CONFLICT" in stderr or "Automatic merge failed" in stderr:
    print(_Red+"合并冲突，请手动解决"+_End)
    sys.exit(1)
  run_git_cmd(["push", "origin", target], cwd=cwd)

  if branch != target:
    run_git_cmd(["switch", branch], cwd=cwd)
    
  print(_Green+f"已将 {source} 合并到 {target} 并推送"+_End)

def update(cwd=None):
  """更新当前分支"""
  if not is_clean(cwd):
    print(_Red+"工作区有未提交更改，请先 stash 或 commit"+_End)
    sys.exit(1)
  
  branch = current_branch(cwd)
  run_git_cmd(["fetch", "origin"], cwd=cwd)
  stdout, stderr = run_git_cmd(["rebase", f"origin/{branch}"], cwd=cwd, allow_fail=True)
  
  if "CONFLICT" in stderr:
    print(_Red+"更新冲突，请手动解决后执行: git rebase --continue"+_End)
    sys.exit(1)
  elif "up to date" in stdout or "up to date" in stderr:
    print(_Green+f"{branch} 已是最新"+_End)
  else:
    print(_Green+f"已更新 {branch}"+_End)

def main():
  parser = argparse.ArgumentParser(description="Git 简易工具")
  parser.add_argument("--action", "-a", choices=["submit", "merge", "update"], required=True)
  parser.add_argument("--src", "-s", help="源分支（merge 时必需）")
  parser.add_argument("--target", "-t", default="main", help="目标分支（默认: main）")
  parser.add_argument("--files", "-f", nargs="+", default=["."], help="提交文件（默认: .）")
  parser.add_argument("--message", "-m", default="Shamrock_PC", help="提交信息")
  parser.add_argument("--repo", "-r", type=Path, default=Path.cwd(), help="仓库路径")

  args = parser.parse_args()
  repo = args.repo.resolve()

  if not (repo / ".git").exists():
    print(_Red+f"'{repo}' 不是 Git 仓库"+_End)
    sys.exit(1)

  try:
    if args.action == "submit":
      submit(args.files, args.message, cwd=repo)
    elif args.action == "merge":
      if not args.src:
        parser.error("--src 是 merge 操作必需的参数")
      merge(args.src, args.target, cwd=repo)
    elif args.action == "update":
      update(cwd=repo)
  except KeyboardInterrupt:
    print(_Red+"\n操作被用户中断"+_End)
    sys.exit(130)
  except Exception as e:
    print(_Red+f"错误: {e}"+_End)
    sys.exit(1)

if __name__ == "__main__":
  main()


#         ,--.                                                 ,--.     
#  ,---.  |  ,---.   ,--,--. ,--,--,--. ,--.--.  ,---.   ,---. |  |,-.  
# (  .-'  |  .-.  | ' ,-.  | |        | |  .--' | .-. | | .--' |     /  
# .-'  `) |  | |  | \ '-'  | |  |  |  | |  |    ' '-' ' \ `--. |  \  \  
# `----'  `--' `--'  `--`--' `--`--`--' `--'     `---'   `---' `--'`--' 

