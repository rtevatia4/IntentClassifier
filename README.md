# 🧠 Intent Classification using ML.NET  
### Classifying Cloud Operations from Natural Language Queries

---

## 🪜 Overview

This project explores how to build a **Machine Learning model** using **ML.NET** to classify user commands or natural language queries into specific **intents** related to **cloud operations** — for example:

| Example Query | Classified Intent |
|----------------|------------------|
| "Start my app instance" | StartApp |
| "Stop all services" | StopApp |
| "Show app details" | GetAppDetails |

The model helps interpret developer or admin inputs and trigger the right backend operation automatically.

---

## ⚙️ ML Model Lifecycle in ML.NET

The machine learning process in ML.NET typically follows these **key stages**:

1. **Data Collection**
2. **Data Preparation**
3. **Feature Engineering**
4. **Model Selection & Training**
5. **Model Evaluation**
6. **Model Saving & Deployment**
7. **Model Consumption (Prediction)**

Let’s walk through each step with examples.

---

## 1️⃣ Data Collection

We first define **training data** — examples of text queries and their corresponding intents.

Example:  
```csv
Text,Intent
"Start my app",StartApp
"Can you bring up the service?",StartApp
"Stop all running jobs",StopApp
"Shut down the app instance",StopApp
"Get app details",GetAppDetails
"Show me info about my app",GetAppDetails
```

Each row represents a **labeled example**.  
ML.NET will learn patterns between the `Text` and the `Intent`.

---

## 2️⃣ Data Preparation

Load the dataset into an ML.NET data view and define the schema:

```csharp
public class IntentInput
{
    public string Text { get; set; }
    public string Intent { get; set; }
}

var mlContext = new MLContext();
var data = mlContext.Data.LoadFromTextFile<IntentInput>(
    "intents.csv", hasHeader: true, separatorChar: ',');
```

We can optionally split the dataset for training and testing:

```csharp
var split = mlContext.Data.TrainTestSplit(data, testFraction: 0.2);
```

---

## 3️⃣ Feature Engineering

Text data needs to be converted into **numerical features** so that the model can understand it.  
ML.NET provides a `FeaturizeText` transformation that automatically handles:

- Tokenization (breaking text into words)
- Normalization (lowercasing, removing punctuation)
- N-gram extraction (learning patterns like “start app”, “stop service”)

```csharp
var pipeline = mlContext.Transforms.Text.FeaturizeText(
                    "Features", nameof(IntentInput.Text))
                .Append(mlContext.Transforms.Conversion.MapValueToKey(
                    "Label", nameof(IntentInput.Intent)));
```

---

## 4️⃣ Model Selection & Training

For **multiclass classification**, a good algorithm is **SDCA Maximum Entropy**.  
This is efficient and works well for text data.

```csharp
var trainer = mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy();
var trainingPipeline = pipeline
    .Append(trainer)
    .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

var model = trainingPipeline.Fit(split.TrainSet);
```

---

## 5️⃣ Model Evaluation

Once trained, we evaluate how accurate the model is using the test dataset:

```csharp
var predictions = model.Transform(split.TestSet);
var metrics = mlContext.MulticlassClassification.Evaluate(predictions);
```

Example output:

```
MicroAccuracy:    0.93
MacroAccuracy:    0.92
LogLoss:          0.18
LogLossReduction: 0.84
```

✅ **Interpretation:**
- Accuracy above 90% indicates good performance.
- LogLoss close to 0 indicates better confidence in predictions.

---

## 6️⃣ Model Saving & Deployment

After validation, save the trained model as a `.zip` file for later use:

```csharp
mlContext.Model.Save(model, data.Schema, "IntentModel.zip");
```

This model can now be:
- Loaded inside a web API or console app
- Packaged as part of a chatbot or automation tool
- Deployed as an endpoint (e.g., REST API)

---

## 7️⃣ Model Consumption (Prediction)

To use the model for real-time predictions:

```csharp
var loadedModel = mlContext.Model.Load("IntentModel.zip", out _);
var predictor = mlContext.Model.CreatePredictionEngine<IntentInput, IntentPrediction>(loadedModel);

public class IntentPrediction
{
    public string PredictedLabel { get; set; }
}

var result = predictor.Predict(new IntentInput { Text = "Restart my app" });
Console.WriteLine($"Predicted Intent: {result.PredictedLabel}");
```

**Output:**
```
Predicted Intent: StartApp
```

---

## 🧩 Example Architecture

```text
+------------------------+
|  User Query (Text)     |
+----------+-------------+
           |
           v
+------------------------+
|  Intent Model (.zip)   |
|  (ML.NET Pipeline)     |
+----------+-------------+
           |
           v
+------------------------+
|  Predicted Intent      |
|  → Triggers Cloud API  |
+------------------------+
```

---

## 🚀 Potential Extensions

- Add **new intents** like `ScaleApp`, `GetLogs`, `DeployApp`.
- Implement **feedback loop** — store misclassifications and retrain periodically.
- Integrate with:
  - **CLI Tools** (PCF CLI, Azure CLI)
  - **ChatOps Bots**
  - **Internal dashboards** for natural-language commands

---

## ✅ Summary

| Stage | What Happens | Example |
|--------|---------------|----------|
| Data Collection | Gather labeled examples | “Start app” → StartApp |
| Data Prep | Load and split data | Train/Test split |
| Feature Engineering | Convert text → vectors | “stop app” → numerical |
| Training | Train ML model | SDCA algorithm |
| Evaluation | Measure accuracy | 93% |
| Deployment | Save model | IntentModel.zip |
| Prediction | Use model for new text | “Show app info” → GetAppDetails |

---

**With ML.NET, we can empower our existing .NET tools with natural language understanding — making automation smarter and more intuitive.**
