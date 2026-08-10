# Streamlit Community Cloud Deployment Guide
### For Batchmates (D&G Project T2)

This step-by-step guide walks you through deploying your Streamlit Machine Learning & RAG web application for free on **Streamlit Community Cloud** using your GitHub repository.

---

## 1. File Placement

Your project files (`app.py` and `requirements.txt`) can be located either at the **root level** or inside a **subfolder** (e.g. `streamlit_deployment/`).

- **Option A (Subfolder):** `streamlit_deployment/app.py` and `streamlit_deployment/requirements.txt`
- **Option B (Root):** `app.py` and `requirements.txt`

*Note: Streamlit Cloud automatically installs packages from whichever folder contains `app.py`.*

---

## 2. Grant Streamlit Access to Private Repositories (CRITICAL STEP)

If your GitHub repository is **Private**, Streamlit Cloud needs permission to read your private repo:

1. Go to [share.streamlit.io](https://share.streamlit.io) and log in with your GitHub account.
2. Click your profile avatar / settings icon (bottom-left corner) → **Settings**.
3. Under **Linked Accounts** / **GitHub Integration**, click **Manage Permissions**.
4. You will be redirected to GitHub permissions:
   - Select **All repositories** (or click **Select repositories** and choose your private project repo).
5. Click **Save / Grant Access**.

---

## 3. Deploy Your Application on Streamlit Cloud

1. Go to: [share.streamlit.io/deploy](https://share.streamlit.io/deploy)
2. Fill in the deployment form:
   - **Repository:** Enter your GitHub repo URL (e.g., `https://github.com/your-username/your-repo-name`)
   - **Branch:** `main`
   - **Main file path:** Enter the path to `app.py`:
     - If inside a subfolder: `streamlit_deployment/app.py`
     - If at root: `app.py`
   - **App URL (optional):** Custom name (e.g., `smart-mcq-solver`)
3. Click **Deploy!**

Streamlit Cloud will automatically locate the `requirements.txt` in the same directory as `app.py`, install the required packages, and launch your application within 1 to 2 minutes!

---

## 4. Verify & Submit Your Live App Link

1. Copy your live Streamlit URL (e.g., `https://your-app-name.streamlit.app`).
2. Open an **Incognito / Private Window** in your browser and paste the link.
3. If the app loads directly without prompting for a login, your live deployment is **Public and Ready for Evaluation**!
4. Paste the `.streamlit.app` link into the official Google Form submission.
