#                           git相关脚本
#                           2025/12/1
#                            shamrock

import subprocess
import os, argparse, sys
from datetime import datetime
from pathlib import Path

#----------------- 要提交的文件 ----------------

submit_files = [
  # '.\\Projects\\',
  # '.\\Tasks\\DeepLearning\\',
  # '.\\Tasks\\DeepLearning\\**\\*.ipynb',
  # '.\\Tasks\\DeepLearning\\**\\*.jpg',
  # '.\\Tasks\\ReinforcementLearning\\**\\*.ipynb',
  # '.\\Tasks\\ReinforcementLearning\\**\\*.jpg',
  # '.\\Tasks\\ReinforcementLearning\\'
  '.'
]

commit_message = 'Shamrock_PC'

#----------------- 工具函数 ----------------

REPO_ROOT = Path.cwd()

def run_git_cmd(args, cwd=None, allow_fail=False):
  """安全执行 git 命令，返回 (stdout, stderr)"""
  if cwd is None:
    cwd = Path.cwd()
  cmd = ["git"] + args
  try:
    result = subprocess.run(
      cmd,
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
    print(f"\033[31m Git failed: {' '.join(cmd)}\033[0m")
    print(f"\033[31mstderr:\033[0m {e.stderr}")
    raise

def ensure_clean_working_tree(cwd=None):
  """检查工作区是否干净，否则退出"""
  stdout, _ = run_git_cmd(["status", "--porcelain"], cwd=cwd)
  if stdout:
    print("\033[33m 工作区有未提交修改！请先 stash 或 commit：\033[0m")
    for line in stdout.splitlines():
      print(f"   {line}")
    sys.exit(1)

def get_current_branch(cwd=None):
  branch, _ = run_git_cmd(["rev-parse", "--abbrev-ref", "HEAD"], cwd=cwd)
  return branch

#----------------- 函数 ----------------

def git_add_commit_push(files, commit_message=None, cwd=None):
  if cwd is None:
    cwd = REPO_ROOT
  if isinstance(files, str):
    files = [files]
  
  print(f"Repo root: {cwd}")
  print(f"Adding files: {files}")
  
  # 1. git add
  run_git_cmd(["add"] + files, cwd=cwd)
  print("\033[32mAdded. √ \033[0m")

  # 2. 检查是否真有变更（防空提交）
  staged = run_git_cmd(["diff", "--name-only", "--cached"], cwd=cwd)
  if not staged:
    print("No changes staged. Skipping commit & push.")
    return

  # 3. git commit
  if not commit_message:
    commit_message = f"[Auto] Submit @ {datetime.now().strftime('%Y-%m-%d %H:%M')}"
  run_git_cmd(["commit", "-m", commit_message], cwd=cwd)
  print("\033[32mCommitted. √ \033[0m")

  # 4. git push
  branch = run_git_cmd(["rev-parse", "--abbrev-ref", "HEAD"], cwd=cwd)
  print(f"Pushing to origin/{branch[0]}...")
  # print(branch)
  run_git_cmd(["push", "origin", branch[0]], cwd=cwd)
  print("\033[32mPushed successfully! √ \033[0m")

def git_action_merge(src_branch, target_branch, cwd=None):
  print(f"准备将 '{src_branch}' 合并到 '{target_branch}'")
  input("\nPress Enter to confirm submission (or Ctrl+C to abort)...")
  
  # 1. 检查分支是否存在
  branches, _ = run_git_cmd(["branch", "--list", "--no-color"], cwd=cwd)
  local_branches = [b.strip('* \n') for b in branches.splitlines()]
  if src_branch not in local_branches:
    raise ValueError(f"\033[31m 源分支 '{src_branch}' 不存在！可用分支：{local_branches}\033[0m")
  if target_branch not in local_branches:
    raise ValueError(f"\033[31m 目标分支 '{target_branch}' 不存在！\033[0m")
  
  # 2. 确保工作区干净
  ensure_clean_working_tree(cwd)

  # 3. 切换到目标分支
  print(f"切换到目标分支: {target_branch}")
  run_git_cmd(["checkout", target_branch], cwd=cwd)

  # 4. 拉取远程最新（避免过期）
  print(f"拉取远程 {target_branch} 最新状态...")
  run_git_cmd(["pull", "origin", target_branch], cwd=cwd)

  # 5. 执行合并
  print(f"执行合并: git merge {src_branch}")
  stdout, stderr = run_git_cmd(
    ["merge", src_branch, "--no-edit"],  # --no-edit 避免打开编辑器
    cwd=cwd,
    allow_fail=True
  )

  if "CONFLICT" in stderr or "Automatic merge failed" in stderr:
    print("\033[31m 合并冲突！请手动解决：\033[0m")
    print(stderr)
    print("\n🔧 解决步骤：")
    print("   1. 编辑冲突文件（查找 <<<<<<<）")
    print("   2. git add <resolved-file>")
    print("   3. git commit")
    sys.exit(1)
  elif stdout or stderr:
    print(f"合并输出: {stdout} {stderr}")
  
  print("\033[32m Merged. √ \033[0m")

def main():
  parser = argparse.ArgumentParser(
    description="Git 自动化工具：提交文件 或 合并分支",
    epilog="示例:\n"
            "  # 提交默认文件\n"
            "  python git_tool.py --action submit\n\n"
            "  # 合并 b1 → main\n"
            "  python git_tool.py --action merge --src b1 --target main",
    formatter_class=argparse.RawDescriptionHelpFormatter
  )

  parser.add_argument(
    "--action", "-a",
    choices=["submit", "merge"],
    required=True,
    help="操作类型：submit（提交文件）或 merge（合并分支）"
  )
  parser.add_argument(
    "--src", "-s",
    help="合并时的源分支（--action merge 必需）"
  )
  parser.add_argument(
    "--target", "-t",
    default="main",
    help="合并时的目标分支，默认 main"
  )
  parser.add_argument(
    "--files", "-f",
    nargs="+",
    default=submit_files,
    help="提交的文件列表（--action submit 时有效），默认: %(default)s"
  )
  parser.add_argument(
    "--message", "-m",
    default=commit_message,
    help="提交信息，默认: %(default)s"
  )
  parser.add_argument(
    "--repo", "-r",
    type=Path,
    default=Path(__file__).parent,
    help="仓库根目录，默认脚本所在目录"
  )

  args = parser.parse_args()

  repo_root = args.repo.resolve()
  if not (repo_root / ".git").exists():
    print(f"\033[31m错误: '{repo_root}' 不是 Git 仓库\033[0m")
    sys.exit(1)

  try:
    if args.action == "submit":
      git_add_commit_push(args.files, args.message, cwd=repo_root)
    elif args.action == "merge":
      if not args.src:
        parser.error("--src 是 --action merge 所必需的")
      git_action_merge(args.src, args.target, cwd=repo_root)
  except Exception as e:
    print(f"\033[31m 操作失败: {e}\033[0m")
    sys.exit(1)

if __name__ == "__main__":
  main()


#         ,--.                                                 ,--.     
#  ,---.  |  ,---.   ,--,--. ,--,--,--. ,--.--.  ,---.   ,---. |  |,-.  
# (  .-'  |  .-.  | ' ,-.  | |        | |  .--' | .-. | | .--' |     /  
# .-'  `) |  | |  | \ '-'  | |  |  |  | |  |    ' '-' ' \ `--. |  \  \  
# `----'  `--' `--'  `--`--' `--`--`--' `--'     `---'   `---' `--'`--' 

