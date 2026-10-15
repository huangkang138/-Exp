$ErrorActionPreference = 'Stop'
$python = 'D:\soft\Anaconda\envs\GB-MACO_python\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    throw "找不到 Python 环境：$python"
}
& $python -B (Join-Path $PSScriptRoot 'run.py') @args
if ($LASTEXITCODE -ne 0) { throw "评估中断，退出码：$LASTEXITCODE" }
