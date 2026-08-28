param(
    [string]$OutputPath = "results/raw/environment.txt"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$resolvedOutput = Join-Path $repoRoot $OutputPath
$outputDirectory = Split-Path -Parent $resolvedOutput
New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null

$lines = [System.Collections.Generic.List[string]]::new()
$lines.Add("captured_at=$(Get-Date -Format o)")
$lines.Add("host=$env:COMPUTERNAME")
$lines.Add("git_commit=$(git -C $repoRoot rev-parse HEAD 2>$null)")
$lines.Add("")

$commands = @(
    @{ Name = "nvidia-smi"; Command = { nvidia-smi } },
    @{ Name = "nvidia-smi-query"; Command = {
        nvidia-smi --query-gpu=name,uuid,driver_version,pstate,clocks.current.graphics,clocks.current.memory,temperature.gpu,power.draw,power.limit --format=csv
    } },
    @{ Name = "nvcc"; Command = { nvcc --version } },
    @{ Name = "python"; Command = { python --version } },
    @{ Name = "python-packages"; Command = {
        python -c "import importlib; names=['torch','triton','vllm']; print('\n'.join(n+'='+getattr(importlib.import_module(n), '__version__', 'unknown') for n in names))"
    } },
    @{ Name = "pip-freeze"; Command = { python -m pip freeze } }
)

foreach ($entry in $commands) {
    $lines.Add("=== $($entry.Name) ===")
    try {
        $result = & $entry.Command 2>&1 | Out-String
        $lines.Add($result.TrimEnd())
    }
    catch {
        $lines.Add("UNAVAILABLE: $($_.Exception.Message)")
    }
    $lines.Add("")
}

$lines | Set-Content -LiteralPath $resolvedOutput -Encoding utf8
Write-Output "Wrote environment manifest to $resolvedOutput"
