import React, { useState } from "react";
import { LineChart, Line, BarChart, Bar, ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,} from "recharts";
import { Upload, TrendingUp, Database, BarChart3, Brain, FileText, Target,} from "lucide-react";

const API_URL = "http://localhost:5000/api";

const PROJECTS = [
  {
    id: 1,
    name: "Housing Price Predictions",
    type: "regression",
    description: "Predict real estate prices using ML regression models",
    icon: "🏘️",
    impact: "High - Real-world application in finance and real estate",
  },
  {
    id: 2,
    name: "Customer Churn Analysis",
    type: "classification",
    description: "Identify customers likely to leave and retention strategies",
    icon: "📊",
    impact: "High - Critical for business retention and revenue",
  },
  {
    id: 3,
    name: "Credit Card Fraud Detection",
    type: "classification",
    description: "Detect fraudulent transactions using anomaly detection",
    icon: "💳",
    impact: "Very High - Financial security and fraud prevention",
  },
  {
    id: 4,
    name: "COVID-19 Analysis & Predictions",
    type: "time_series",
    description: "Analyze pandemic trends and predict future cases",
    icon: "🦠",
    impact: "Very High - Public health and policy decisions",
  },
  {
    id: 5,
    name: "HR Analytics & Performance",
    type: "business",
    description: "Track employee performance and predict attrition",
    icon: "👥",
    impact: "High - Optimize workforce and reduce turnover costs",
  },
];

export default function DataAnalysisPlatform() {
  const [selectedProject, setSelectedProject] = useState(null);
  const [file, setFile] = useState(null);
  const [analysisData, setAnalysisData] = useState(null);
  const [modelResults, setModelResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState("projects");

  const handleFileUpload = (event) => {
    const uploadedFile = event.target.files[0];
    setFile(uploadedFile);
  };

  const analyzeData = async () => {
    if (!file) {
      alert("Please upload a CSV file first");
      return;
    }

    if (!selectedProject) {
      alert("Please select a project");
      return;
    }

    setLoading(true);
    const formData = new FormData();
    formData.append("file", file);
    formData.append("project_type", selectedProject.type);

    try {
      const response = await fetch(`${API_URL}/analyze-dataset`, {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      setAnalysisData(data);
      setActiveTab("analysis");
    } catch (error) {
      // keep error message simple
      alert(
        "Error analyzing data. Make sure Flask server is running on port 5000."
      );
      console.error(error);
    }
    setLoading(false);
  };

  const trainModel = async () => {
    if (!analysisData || !selectedProject) return;

    setLoading(true);
    try {
      const targetColumn =
        analysisData.column_names[analysisData.column_names.length - 1];
      const response = await fetch(`${API_URL}/train-model`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          target_column: targetColumn,
          model_type: "random_forest",
          task_type: selectedProject.type === "regression" ? "regression" : "classification",
        }),
      });
      const data = await response.json();
      setModelResults(data);
      setActiveTab("model");
    } catch (error) {
      alert("Error training model");
      console.error(error);
    }
    setLoading(false);
  };

  return (
    <div className="page_container">
      <div className="page_inner">
        {/* Header */}
        <div className="header_section">
          <div className="header_top">
            <Brain className="icon_brain" />
            <h1 className="title_text">ML Data Analytics</h1>
          </div>
          <p className="subtitle_text">Enterprise-Grade Machine Learning Analysis Platform</p>
          <p className="muted_text">5 High-Impact Use Cases • Real-World Applications</p>
        </div>

        {/* Tabs */}
        <div className="tabs_row">
          {["projects", "analysis", "model"].map((tab) => {
            const disabled =
              (tab === "analysis" && !analysisData) || (tab === "model" && !modelResults);
            const active = activeTab === tab;
            return (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                disabled={disabled}
                className={`tab_button ${active ? "tab_button_active" : "tab_button_inactive"}`}
              >
                {tab.charAt(0).toUpperCase() + tab.slice(1)}
              </button>
            );
          })}
        </div>

        {/* Projects */}
        {activeTab === "projects" && (
          <div className="projects_grid">
            {PROJECTS.map((project) => {
              const isSelected = selectedProject?.id === project.id;
              return (
                <div
                  key={project.id}
                  onClick={() => {
                    setSelectedProject(project);
                    setActiveTab("analysis");
                    setAnalysisData(null);
                    setModelResults(null);
                    setFile(null);
                  }}
                  className={`project_card ${isSelected ? "project_card_selected" : ""}`}
                >
                  <div className="project_icon">{project.icon}</div>
                  <h3 className="project_title">{project.name}</h3>
                  <p className="project_description">{project.description}</p>

                  <div className="project_meta">
                    <span className="project_tag">{project.type}</span>
                    <span className="project_impact">⚡ {project.impact.split(" - ")[0]}</span>
                  </div>
                  <p className="project_impact_detail">{project.impact.split(" - ")[1]}</p>
                </div>
              );
            })}
          </div>
        )}

        {/* Analysis tab */}
        {activeTab === "analysis" && selectedProject && (
          <div className="analysis_section">
            {!analysisData && (
              <div className="upload_panel">
                <div className="upload_header">
                  <div className="project_icon_large">{selectedProject.icon}</div>
                  <div>
                    <h2 className="upload_title">{selectedProject.name}</h2>
                    <p className="upload_sub">{selectedProject.description}</p>
                  </div>
                </div>

                <div className="upload_box">
                  <Upload className="icon_upload" />
                  <p className="upload_prompt">Upload Your Dataset</p>
                  <p className="upload_hint">CSV or Excel files supported</p>

                  <input
                    id="file-upload"
                    type="file"
                    accept=".csv,.xlsx,.xls"
                    onChange={handleFileUpload}
                    className="file_input"
                  />

                  <label htmlFor="file-upload" className="choose_button">Choose File</label>

                  {file && (
                    <p className="file_selected">
                      <span>✓</span> {file.name}
                    </p>
                  )}
                </div>

                <button
                  onClick={analyzeData}
                  disabled={!file || loading}
                  className="analyze_button"
                >
                  <BarChart3 className="icon_btn" />
                  {loading ? "Analyzing Dataset..." : "Analyze Dataset"}
                </button>
              </div>
            )}

            {analysisData && (
              <div className="analysis_results">
                <div className="overview_card">
                  <h2 className="overview_title">
                    <Database className="icon_db" /> Dataset Overview
                  </h2>

                  <div className="overview_grid">
                    <div className="overview_item blue_item">
                      <div className="overview_value">{analysisData.shape?.rows ?? "N/A"}</div>
                      <div className="overview_label">Total Rows</div>
                    </div>

                    <div className="overview_item purple_item">
                      <div className="overview_value">{analysisData.shape?.columns ?? "N/A"}</div>
                      <div className="overview_label">Features</div>
                    </div>

                    <div className="overview_item green_item">
                      <div className="overview_value">{analysisData.numeric_cols ?? "N/A"}</div>
                      <div className="overview_label">Numeric</div>
                    </div>

                    <div className="overview_item pink_item">
                      <div className="overview_value">
                        {analysisData.data_quality ? `${analysisData.data_quality.completeness.toFixed(1)}%` : "N/A"}
                      </div>
                      <div className="overview_label">Complete</div>
                    </div>
                  </div>
                </div>

                {/* Distributions */}
                {analysisData.distributions && (
                  <div className="chart_card">
                    <h3 className="chart_title">Feature Distributions</h3>
                    <ResponsiveContainer width="100%" height={400}>
                      <BarChart data={analysisData.distributions}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#4a5568" />
                        <XAxis dataKey="feature" stroke="#e2e8f0" />
                        <YAxis stroke="#e2e8f0" />
                        <Tooltip contentStyle={{ backgroundColor: "#1e293b", border: "1px solid #6366f1" }} />
                        <Legend />
                        <Bar dataKey="mean" fill="#8b5cf6" name="Mean Value" />
                        <Bar dataKey="std" fill="#ec4899" name="Std Dev" />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                )}

                {/* Correlations */}
                {analysisData.correlations && (
                  <div className="chart_card">
                    <h3 className="chart_title">Top Correlations</h3>
                    <ResponsiveContainer width="100%" height={400}>
                      <BarChart data={analysisData.correlations.slice(0, 10)} layout="vertical">
                        <CartesianGrid strokeDasharray="3 3" stroke="#4a5568" />
                        <XAxis type="number" domain={[-1, 1]} stroke="#e2e8f0" />
                        <YAxis type="category" dataKey="pair" width={200} stroke="#e2e8f0" />
                        <Tooltip contentStyle={{ backgroundColor: "#1e293b", border: "1px solid #6366f1" }} />
                        <Bar dataKey="correlation" fill="#8b5cf6" />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                )}

                <button
                  onClick={trainModel}
                  disabled={loading}
                  className="train_button"
                >
                  <Target className="icon_btn" />
                  {loading ? "Training Model..." : "Train ML Model"}
                </button>
              </div>
            )}
          </div>
        )}

        {/* Model tab */}
        {activeTab === "model" && modelResults && (
          <div className="model_section">
            <div className="model_card">
              <h2 className="model_title"><TrendingUp className="icon_trend" /> Model Performance</h2>

              <div className="model_grid">
                {modelResults.accuracy && (
                  <div className="metric_card green_metric">
                    <div className="metric_value">{(modelResults.accuracy * 100).toFixed(2)}%</div>
                    <div className="metric_label">Accuracy</div>
                  </div>
                )}

                {modelResults.r2_score && (
                  <div className="metric_card blue_metric">
                    <div className="metric_value">{modelResults.r2_score.toFixed(4)}</div>
                    <div className="metric_label">R² Score</div>
                  </div>
                )}

                {modelResults.rmse && (
                  <div className="metric_card purple_metric">
                    <div className="metric_value">{modelResults.rmse.toFixed(2)}</div>
                    <div className="metric_label">RMSE</div>
                  </div>
                )}

                <div className="metric_card pink_metric">
                  <div className="metric_value">{(modelResults.model || "N/A").replace("_", " ").toUpperCase()}</div>
                  <div className="metric_label">Model Type</div>
                </div>
              </div>
            </div>

            {modelResults.feature_importance && (
              <div className="chart_card">
                <h3 className="chart_title">Feature Importance</h3>
                <ResponsiveContainer width="100%" height={400}>
                  <BarChart data={modelResults.feature_importance.slice(0, 10)}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#4a5568" />
                    <XAxis dataKey="feature" stroke="#e2e8f0" angle={-45} textAnchor="end" height={100} />
                    <YAxis stroke="#e2e8f0" />
                    <Tooltip contentStyle={{ backgroundColor: "#1e293b", border: "1px solid #6366f1" }} />
                    <Bar dataKey="importance" fill="#8b5cf6" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}

            <div className="complete_card">
              <h2 className="complete_title">🎉 Analysis Complete!</h2>
              <p className="complete_sub">Successfully analyzed {selectedProject?.name}</p>
              <button
                className="again_button"
                onClick={() => {
                  setActiveTab("projects");
                  setSelectedProject(null);
                  setAnalysisData(null);
                  setModelResults(null);
                  setFile(null);
                }}
              >
                Analyze Another Dataset
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
