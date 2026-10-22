# schedule_sentinel_task.ps1
# Schedules the 5-Second Sentinel to auto-start at 09:10 AM Mon-Fri via Windows Task Scheduler

$Action = New-ScheduledTaskAction -Execute "C:\Users\manoj\scout\nifty_options_predictor\run_sentinel.bat"
$Trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday, Tuesday, Wednesday, Thursday, Friday -At 9:10AM
$Settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -WakeToRun -StartWhenAvailable
$TaskName = "NiftySensexOptionsSentinel"

# Register the task
Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Settings $Settings -Description "Auto-starts Nifty/Sensex 5-second live sentinel at 09:10 AM on trading days" -Force
Write-Host "✅ Scheduled Task '$TaskName' registered successfully to start every Mon-Fri at 09:10 AM!" -ForegroundColor Green
