# Launch the validated CPU experiment from any PowerShell working directory.
$python = 'D:\soft\Anaconda\envs\GB-MACO_python\python.exe'
$entry = Join-Path $PSScriptRoot 'run_opt_b.py'
if (-not (Test-Path -LiteralPath $python)) {
    throw "Python environment not found: $python"
}
& $python $entry @args
if ($LASTEXITCODE -ne 0) {
    throw "Opt-B experiment failed with exit code $LASTEXITCODE"
}
