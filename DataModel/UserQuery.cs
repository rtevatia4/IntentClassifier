using Microsoft.ML.Data;

namespace IntentClassifier.DataModel
{
    public class UserQuery
    {
        [LoadColumn(0)]
        public string Text { get; set; }

        [LoadColumn(1)]
        public string Intent { get; set; }
    }
}
