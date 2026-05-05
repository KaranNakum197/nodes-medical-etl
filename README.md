# 🏥 Nodes of the Trench: Medical ETL Pipeline

![AMD Developer Cloud](https://img.shields.io/badge/AMD-Developer_Cloud-black?logo=amd)
![ROCm](https://img.shields.io/badge/Powered_by-ROCm-red)
![CrewAI](https://img.shields.io/badge/Agents-CrewAI-orange)

Built for the **AMD Developer Hackathon**, this project is an end-to-end, agentic ETL (Extract, Transform, Load) pipeline designed to digitize massive volumes of unstructured medical laboratory reports. 

## 🚀 The Challenge
Extracting data from complex medical PDFs is notoriously difficult due to dense text, tables, and inconsistent formatting. Standard OCR fails, and metadata is unreliable. We built a system capable of handling high-throughput medical document extraction securely and accurately.

## 🧠 The Solution
We leverage the massive memory bandwidth of **AMD Instinct MI300X** instances to run **Qwen2-VL-7B**, extracting pure JSON from raw document images. This data is then routed through a **CrewAI** agentic workflow for validation, formatting, and anomaly detection before being securely committed to a **PostgreSQL** database.

### 🛠️ Tech Stack
*   **Hardware:** AMD Instinct MI300X (via AMD Developer Cloud)
*   **Core Compute Stack:** ROCm, PyTorch
*   **Vision Model (Track 3):** Qwen2-VL-7B-Instruct
*   **Agentic Framework (Track 1):** CrewAI, LangChain
*   **Database:** PostgreSQL
*   **Frontend:** React / FastAPI

## 🚢 Build in Public
We are building this live! Check out our technical updates and ROCm developer feedback on social media:
*   [Update 1: Taming the MI300X] (Link to be added)
*   [Update 2: Agentic Orchestration] (Link to be added)

## 💻 Local Setup
*(Instructions for running the FastAPI backend and CrewAI agents will be added here as we build).*
