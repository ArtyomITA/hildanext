$cutoff = [datetime]"2026-03-02T00:00:00"
$events = Get-WinEvent -LogName System -MaxEvents 5000
$filtered = $events | Where-Object {
    $_.TimeCreated -ge $cutoff -and
    ($_.Level -le 2 -or $_.Id -eq 41 -or $_.Id -eq 6008 -or $_.Id -eq 1074 -or $_.Id -eq 6006 -or $_.Id -eq 6013)
}
foreach ($e in $filtered) {
    $msg = $e.Message -replace "`r`n"," " -replace "`n"," "
    if ($msg.Length -gt 200) { $msg = $msg.Substring(0,200) }
    Write-Output ($e.TimeCreated.ToString("yyyy-MM-dd HH:mm:ss") + " | ID=" + $e.Id + " | Lv=" + $e.Level + " | " + $msg)
}
