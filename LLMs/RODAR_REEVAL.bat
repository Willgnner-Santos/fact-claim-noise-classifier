@echo off
REM ============================================================
REM Re-evaluate LLMs on the canonical test set (BERT) - Fact/Claim/Noise
REM Resumable: if interrupted, re-run to continue from where it stopped.
REM ============================================================
cd /d "E:\Anonymous\Fato_Tese_Ruido\LLMs"
set PYTHONIOENCODING=utf-8

echo.
echo Starting re-evaluation (all models x all strategies)...
echo Full log at: data\reeval_run.log
echo.

venv\Scripts\python.exe scripts\04_reeval_on_bert_test.py --models all --strategies all --concurrency 8

echo.
echo ============================================================
echo DONE. Results at: data\results_bertset\summary_berttest.csv
echo ============================================================
pause
