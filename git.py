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
  '''
    将修改上传到远程仓库
  '''
  if cwd is None:
    cwd = REPO_ROOT
  if isinstance(files, str):
    files = [files]
  
  print(f"Repo root: {cwd}")
  print(f"Adding files: {files}")
  
  # 1. git add
  run_git_cmd(["add"] + files, cwd=cwd)
  print("\033[34m Added. \033[0m")

  # 2. 检查是否真有变更（防空提交）
  staged = run_git_cmd(["diff", "--name-only", "--cached"], cwd=cwd)
  if not staged:
    print("\033[33m No changes staged. Skipping commit & push. \033[0m")
    return

  # 3. git commit
  if not commit_message:
    commit_message = f"[Auto] Submit @ {datetime.now().strftime('%Y-%m-%d %H:%M')}"
  run_git_cmd(["commit", "-m", commit_message], cwd=cwd)
  print("\033[34m Committed. \033[0m")

  # 4. git push
  branch = run_git_cmd(["rev-parse", "--abbrev-ref", "HEAD"], cwd=cwd)
  print(f"\033[34m Pushing to origin/{branch[0]}... \033[0m")
  run_git_cmd(["push", "origin", branch[0]], cwd=cwd)
  print("\033[32m Pushed successfully! √ \033[0m")

def git_merge(src_branch, target_branch, cwd=None):
  '''
    将src_branch合并到target_branch
  '''
  original_branch, _ = run_git_cmd(["rev-parse", "--abbrev-ref", "HEAD"], cwd=cwd)
  print(f"\033[34m Current branch: '{original_branch}' \033[0m")
  print(f"\033[34m Preparing to merge '{src_branch}' into '{target_branch}' \033[0m")
  input("\nPress Enter to confirm submission (or Ctrl+C to abort)...")
  
  # 1. 检查分支是否存在
  branches, _ = run_git_cmd(["branch", "--list", "--no-color"], cwd=cwd)
  local_branches = [b.strip('* \n') for b in branches.splitlines()]
  if src_branch not in local_branches:
    raise ValueError(f"\033[31mError: Source branch '{src_branch}' does not exist! Available branches: {local_branches}\033[0m  ")
  if target_branch not in local_branches:
    raise ValueError(f"\033[31mError: Target branch '{target_branch}' does not exist!\033[0m  ")
  
  # 2. 确保工作区干净
  ensure_clean_working_tree(cwd)

  try:
    # 3. 切换到目标分支
    print(f"\033[34m Switching to target branch: {target_branch} \033[0m")
    run_git_cmd(["checkout", target_branch], cwd=cwd)

    # 4. 拉取远程最新（避免过期）
    print(f"\033[34m Fetching latest updates for remote '{target_branch}'... \033[0m")
    run_git_cmd(["pull", "origin", target_branch], cwd=cwd)

    # 5. 执行合并
    print(f"\033[34m Executing merge: git merge {src_branch} \033[0m")
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

      original_branch, _ = run_git_cmd(["rev-parse", "--abbrev-ref", "HEAD"], cwd=cwd)
      print(f"\033[34m Current branch: '{original_branch}' \033[0m")
      return
    elif stdout or stderr:
      print(f"{stdout} {stderr}")
      print("\033[32m Merged. √ \033[0m")
  finally:
    if original_branch != target_branch:
      print(f"\033[36m → Switching back to original branch: '{original_branch}' \033[0m")
      try:
        run_git_cmd(["checkout", original_branch], cwd=cwd)
        print(f"\033[32m Back on '{original_branch}' ✔ \033[0m")
      except Exception as e:
        print(f"\033[33m ⚠ Warning: Failed to switch back to '{original_branch}': {e}\033[0m")

def git_update(cwd=None):
  '''
    更新当前分支
  '''
  current_branch, err = run_git_cmd(["rev-parse", "--abbrev-ref", "HEAD"], cwd=cwd)
  print(f"\033[34m Current branch: '{current_branch}' \033[0m")

  ensure_clean_working_tree(cwd)

  print("\033[34m Fetching remote updates... \033[0m")
  run_git_cmd(["fetch", "origin"], cwd=cwd)

  remote_ref = f"origin/{current_branch}"
  print(f"\033[34m Rebasing onto {remote_ref}... \033[0m")
  stdout, stderr = run_git_cmd(
    ["rebase", remote_ref],
    cwd=cwd,
    allow_fail=True
  )

  if "CONFLICT" in stderr or "rebase in progress" in stderr:
    print("\033[33m Rebase paused due to conflicts.\033[0m")
    print("   Please resolve conflicts, then run:")
    print("      git rebase --continue")
    print("   Or abort with:")
    print("      git rebase --abort")
    return False
  elif "up to date" in stdout or "up to date" in stderr:
    print("\033[32m Already up to date. √ \033[0m")
    return True
  elif "Fast-forwarded" in stdout or "Successfully rebased" in stdout:
    print(f"\033[32m Successfully updated √ '{current_branch}'!\033[0m")
    return True
  else:
    print(f"\033[31m Update failed:\033[0m {stderr or stdout}")
    return False

def main():
  parser = argparse.ArgumentParser(
    description="Git 自动化工具：提交文件、合并分支 或 更新当前分支",
    epilog="示例:\n"
            "  # 提交默认文件\n"
            "  python git_tool.py --action submit\n\n"
            "  # 合并 b1 → main\n"
            "  python git_tool.py --action merge --src b1 --target main\n\n"
            "  # 更新当前分支（fetch + rebase）\n"
            "  python git_tool.py --action update",
    formatter_class=argparse.RawDescriptionHelpFormatter
  )

  parser.add_argument(
    "--action", "-a",
    choices=["submit", "merge", "update"],
    required=True,
    help="操作类型：submit（提交文件）、merge（合并分支）或 update（更新当前分支）"
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
    print(f"\033[31m❌ 错误: '{repo_root}' 不是 Git 仓库\033[0m")
    sys.exit(1)

  try:
    if args.action == "submit":
      git_add_commit_push(args.files, args.message, cwd=repo_root)
    elif args.action == "merge":
      if not args.src:
        parser.error("--src 是 --action merge 所必需的")
      git_merge(args.src, args.target, cwd=repo_root)
    elif args.action == "update":  # ✅ 新增 update 分支
      success = git_update(cwd=repo_root)
      if not success:
        sys.exit(1)  # 更新失败退出
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

