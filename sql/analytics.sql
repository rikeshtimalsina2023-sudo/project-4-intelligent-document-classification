SELECT COUNT(*) AS total_predictions FROM predictions;
SELECT AVG(correct) AS accuracy FROM predictions;
SELECT actual_label, COUNT(*) AS total, SUM(correct) AS correct,
       AVG(correct) AS class_accuracy
FROM predictions GROUP BY actual_label ORDER BY total DESC;
SELECT actual_label, predicted_label, COUNT(*) AS errors
FROM predictions WHERE correct = 0
GROUP BY actual_label, predicted_label ORDER BY errors DESC;
