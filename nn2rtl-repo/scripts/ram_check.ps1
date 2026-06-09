# RAM guard helper (overnight 2026-06-09). Prints "RAM used=NN.N% free=NN.NGB total=NN.NGB"
# Exit code 0 if used < 95%, 1 if used >= 95% (the hard-stop guard).
$os = Get-CimInstance Win32_OperatingSystem
$totKB = $os.TotalVisibleMemorySize
$freeKB = $os.FreePhysicalMemory
$tot = [math]::Round($totKB/1MB,1)
$free = [math]::Round($freeKB/1MB,1)
$used = [math]::Round(($totKB-$freeKB)/$totKB*100,1)
Write-Output ("RAM used=" + $used + "% free=" + $free + "GB total=" + $tot + "GB")
if ($used -ge 95) { exit 1 } else { exit 0 }
