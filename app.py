import os 
import pandas as pd 
from flask import Flask, render_template, request
from src.pipeline.predict_pipeline import PredictPipeline

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get("file")

        if file is None or file.filename == "":
            return render_template("index.html", error="Please upload a valid CSV file.")
        
        try:
            df = pd.read_csv(file)

            pipeline = PredictPipeline()
            preds = pipeline.predict(df)

            # Attach prediction to df
            result_df = df.copy()
            result_df['Prediction'] = preds

            # Count BENIGN / DDoS
            counts = result_df['Prediction'].value_counts().to_dict()

            # Preview table (first 50)
            preview = result_df.head(50).to_dict(orient='records')

            return render_template(
                "index.html",
                results=preds[:20],              # first 20 predictions for list
                predictions_preview=preview,     # table view
                counts=counts,
                success="Prediction completed successfully!"
            )

        except Exception as e:
            return render_template("index.html", error=f"Error: {str(e)}")

    return render_template("index.html")


if __name__ == '__main__':
    app.run(debug=True)
