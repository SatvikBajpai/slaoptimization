# Deploying the Dark Store Order Processing Simulator

This document outlines several methods to deploy the application online.

## Option 1: Streamlit Cloud (Recommended)

Streamlit Cloud is the easiest way to deploy Streamlit applications with minimal setup.

### Steps:

1. Ensure your code is pushed to GitHub (already done at https://github.com/SatvikBajpai/slaoptimization.git)

2. Visit [Streamlit Cloud](https://streamlit.io/cloud) and sign in with your GitHub account

3. Click "New app" button in the dashboard

4. Connect your GitHub repository:
   - Select repository: `SatvikBajpai/slaoptimization`
   - Branch: `main`
   - Main file path: `streamlit_app.py`

5. Click "Deploy" and wait a few minutes

6. Your app will be available at a URL like: `https://slaoptimization-streamlit-app-satvik.streamlit.app`

### Benefits:
- Free tier available (limitations may apply)
- Automatic updates when you push to GitHub
- No infrastructure management required
- Shareable public URL

## Option 2: Render

Render provides a straightforward platform for hosting web services with a generous free tier.

### Steps:

1. Create a new file called `render.yaml` in your repository:

```yaml
services:
  - type: web
    name: sla-optimization
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: streamlit run streamlit_app.py --server.port $PORT --server.address 0.0.0.0
    envVars:
      - key: PYTHON_VERSION
        value: 3.9.0
```

2. Sign up at [Render](https://render.com) and connect your GitHub account

3. Click "New+" and select "Blueprint"

4. Select your repository and follow the setup prompts

5. Your app will be deployed and available at a Render URL

## Option 3: Heroku

Heroku is a popular platform for deploying web applications.

### Steps:

1. Create a `Procfile` in your repository:

```
web: streamlit run streamlit_app.py --server.port $PORT --server.enableCORS false --server.enableXsrfProtection false
```

2. Create a `runtime.txt` file:

```
python-3.9.0
```

3. Install the Heroku CLI and login:

```bash
brew tap heroku/brew && brew install heroku
heroku login
```

4. Create a new Heroku app and deploy:

```bash
cd /path/to/your/repository
heroku create sla-optimization-app
git push heroku main
```

5. Open your app:

```bash
heroku open
```

## Option 4: Docker and Google Cloud Run

For more advanced users who want more control and scalability.

### Steps:

1. Create a `Dockerfile` in your repository:

```Dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

2. Build and test your Docker image locally:

```bash
docker build -t sla-optimization .
docker run -p 8501:8501 sla-optimization
```

3. Deploy to Google Cloud Run:
   - Set up a Google Cloud account and install the Google Cloud SDK
   - Build and push your container
   - Deploy it to Cloud Run

```bash
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/sla-optimization
gcloud run deploy --image gcr.io/YOUR_PROJECT_ID/sla-optimization --platform managed
```

## Additional Considerations

- **Authentication**: If your app contains sensitive data, consider adding authentication
- **Environment Variables**: For production deployments, use environment variables for any configurations
- **Database**: For persistent storage, consider connecting to a database service
- **Monitoring**: Set up monitoring and alerts for production deployments
