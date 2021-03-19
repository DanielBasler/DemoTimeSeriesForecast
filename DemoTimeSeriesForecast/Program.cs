using System;
using Microsoft.ML.Data;
using Microsoft.ML;
using Microsoft.ML.Transforms.TimeSeries;

namespace DemoTimeSeriesForecast
{
    class Program
    {
        static void Main(string[] args)
        {
            var context = new MLContext();

            var data = context.Data.LoadFromTextFile<ModelInput>($"C:/temp/Zeitdaten.csv",
                hasHeader: true, separatorChar: ';');

            var pipeline = context.Forecasting.ForecastBySsa(
                nameof(ModelOutput.ForecastedEnergy),
                nameof(ModelInput.EnergyDemand),
                windowSize: 7,
                seriesLength: 30,
                trainSize: 365,
                horizon: 4);

            var model = pipeline.Fit(data);

            var forecastingEngine = model.CreateTimeSeriesEngine<ModelInput, ModelOutput>(context);

            var forecasts = forecastingEngine.Predict();

            Console.WriteLine("Energie-Prognose");
            Console.WriteLine("----------------");
            foreach (var forecast in forecasts.ForecastedEnergy)
            {
                Console.WriteLine(forecast);
            }
        }
    }

    public class ModelInput
    {
        [LoadColumn(0)]
        public DateTime Date { get; set; }

        [LoadColumn(1)]
        public float EnergyDemand { get; set; }
    }

    public class ModelOutput
    {
        public float[] ForecastedEnergy { get; set; }
    }

}
