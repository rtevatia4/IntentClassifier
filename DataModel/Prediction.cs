using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IntentClassifier.DataModel
{
    public class Prediction
    {
        // map to PredictedLabel column
        [ColumnName("PredictedLabel")]
        public string Intent { get; set; }

        // scores for each class
        public float[] Score { get; set; }
    }
}
