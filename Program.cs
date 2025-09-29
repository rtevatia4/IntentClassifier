using IntentClassifier.DataModel;
using Microsoft.ML;
using System;
using System.IO;
using System.Linq;

class Program
{
    static void Main()
    {
        var mlContext = new MLContext();

        // Load data
        var data = mlContext.Data.LoadFromTextFile<UserQuery>(
            path: "intents.csv",
            hasHeader: true,
            separatorChar: ',');

        // Build pipeline
        var pipeline = mlContext.Transforms.Conversion.MapValueToKey("Label", nameof(UserQuery.Intent))
            .Append(mlContext.Transforms.Text.FeaturizeText("Features", nameof(UserQuery.Text)))
            .Append(mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features"))
            .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

        // Train model
        var model = pipeline.Fit(data);

        // Prediction engine
        var predEngine = mlContext.Model.CreatePredictionEngine<UserQuery, Prediction>(model);

        Console.WriteLine("=== PCF Intent Classifier (Interactive Mode) ===");
        Console.WriteLine("Type a query (or 'exit' to quit):\n");

        while (true)
        {
            Console.Write("User: ");
            var inputText = Console.ReadLine();

            if (string.IsNullOrWhiteSpace(inputText) || inputText.Equals("exit", StringComparison.OrdinalIgnoreCase))
                break;

            var input = new UserQuery { Text = inputText };
            var result = predEngine.Predict(input);

            // Confidence scores
            var maxScore = result.Score.Max();
            var predictedIntent = result.Intent;

            Console.WriteLine($"🤖 Predicted Intent: {predictedIntent}");
            Console.WriteLine($"📊 Confidence: {maxScore:P2}\n");

            // Log low confidence queries for active learning
            if (maxScore < 0.6f)
            {
                File.AppendAllText("low_confidence.log", $"{DateTime.UtcNow}\t{inputText}\t{predictedIntent}\t{maxScore:F2}\n");
                Console.WriteLine("⚠️ Low confidence — logged for review.\n");
            }
        }
    }
}




//using System;
//using System.IO;
//using System.Linq;
//using IntentClassifier.DataModel;
//using Microsoft.ML;
//using Microsoft.ML.Data;

//class Program
//{
//    static void Main()
//    {
//        var mlContext = new MLContext(seed: 0);

//        // 1) Load data
//        var data = mlContext.Data.LoadFromTextFile<UserQuery>(
//            path: "intents.csv",
//            hasHeader: true,
//            separatorChar: ',');

//        // 2) Fit MapValueToKey to create label->key mapping and metadata
//        var mapEstimator = mlContext.Transforms.Conversion.MapValueToKey(
//            outputColumnName: "Label", inputColumnName: nameof(UserQuery.Intent));
//        var mapTransformer = mapEstimator.Fit(data);

//        // 3) Transform training data (now has Label key column & metadata)
//        var mappedData = mapTransformer.Transform(data);

//        // 4) Extract labels (KeyValues / SlotNames) in a version-robust way
//        var labels = ExtractLabelsFromColumn(mappedData.Schema, "Label");

//        if (labels == null || labels.Length == 0)
//        {
//            Console.WriteLine("⚠️ Warning: could not read KeyValues metadata. Falling back to distinct labels from training data.");
//            labels = mlContext.Data.CreateEnumerable<UserQuery>(data, reuseRowObject: false)
//                                   .Select(r => r.Intent)
//                                   .Distinct()
//                                   .ToArray();
//        }

//        // 5) Build pipeline (Featurize -> Trainer using Label -> MapKeyToValue PredictedLabel)
//        var pipeline = mlContext.Transforms.Conversion.MapValueToKey(
//                    outputColumnName: "Label", inputColumnName: nameof(UserQuery.Intent))
//               .Append(mlContext.Transforms.Text.FeaturizeText(
//                    outputColumnName: "Features", inputColumnName: nameof(UserQuery.Text)))
//               .Append(mlContext.MulticlassClassification.Trainers
//                    .SdcaMaximumEntropy(labelColumnName: "Label", featureColumnName: "Features"))
//               .Append(mlContext.Transforms.Conversion.MapKeyToValue(
//                    outputColumnName: "PredictedLabel", inputColumnName: "Label"));


//        // 6) Train on mappedData (contains Label)
//        var model = pipeline.Fit(mappedData);

//        // 7) Create prediction engine
//        var predEngine = mlContext.Model.CreatePredictionEngine<UserQuery, Prediction>(model);

//        Console.WriteLine("=== PCF Intent Classifier (Interactive) ===");
//        Console.WriteLine("Type a query (or 'exit' to quit):\n");

//        while (true)
//        {
//            Console.Write("User: ");
//            var inputText = Console.ReadLine();
//            if (string.IsNullOrWhiteSpace(inputText) ||
//                inputText.Equals("exit", StringComparison.OrdinalIgnoreCase))
//                break;

//            var input = new UserQuery { Text = inputText };
//            var result = predEngine.Predict(input);

//            Console.WriteLine($"\n🤖 Predicted Intent: {result.Intent}");

//            // Print confidence scores paired with labels
//            if (labels != null && result.Score != null && labels.Length == result.Score.Length)
//            {
//                Console.WriteLine("Confidence Scores (sorted):");
//                var pairs = labels.Select((lbl, idx) => new { Label = lbl, Score = result.Score[idx] })
//                                  .OrderByDescending(x => x.Score);
//                foreach (var p in pairs)
//                    Console.WriteLine($"   {p.Label}: {p.Score:P2}");
//            }
//            else
//            {
//                Console.WriteLine("Confidence scores unavailable or mismatch between label count and score length.");
//                Console.WriteLine($"labels.Length = {labels?.Length ?? 0}, scores.Length = {result.Score?.Length ?? 0}");
//            }

//            // Low-confidence logging
//            float maxScore = result.Score?.Max() ?? 0f;
//            if (maxScore < 0.6f)
//            {
//                LogLowConfidence(inputText, result.PredictedLabel, maxScore);
//                Console.WriteLine("\n⚠️ Low confidence — logged for review.\n");
//            }
//            else
//            {
//                Console.WriteLine();
//            }
//        }
//    }

//    // Robust label extraction: iterate annotations.Schema and use GetValue with ref
//    static string[] ExtractLabelsFromColumn(DataViewSchema schema, string columnName)
//    {
//        if (schema.GetColumnOrNull(columnName) is var col && col.HasValue)
//        {
//            var annotations = col.Value.Annotations;

//            // First try "KeyValues"
//            VBuffer<ReadOnlyMemory<char>> keyValues = default;
//            if (annotations.Schema.Any(c => c.Name == "KeyValues"))
//            {
//                annotations.GetValue("KeyValues", ref keyValues);
//                return keyValues.DenseValues().Select(x => x.ToString()).ToArray();
//            }

//            // Fallback: try "SlotNames"
//            VBuffer<ReadOnlyMemory<char>> slotNames = default;
//            if (annotations.Schema.Any(c => c.Name == "SlotNames"))
//            {
//                annotations.GetValue("SlotNames", ref slotNames);
//                return slotNames.DenseValues().Select(x => x.ToString()).ToArray();
//            }
//        }
//        return null;
//    }


//    static void LogLowConfidence(string text, string predictedIntent, float confidence)
//    {
//        var line = $"{DateTime.UtcNow:u} | Input: \"{text}\" | Predicted: {predictedIntent} | Confidence: {confidence:P2}";
//        File.AppendAllLines("low_confidence.log", new[] { line });
//    }
//}
