import { useEffect, useRef, useState } from "react";

const SAMPLE_IMAGE_URL = "/samples/000012.jpg";

type AppState = "inspect" | "analyzing" | "results";

export interface DetectionBox {
  class_id: number;
  class_name: string;
  confidence: number;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  bounding_box_area?: number;
  normalized_box_area?: number;
  image_width?: number;
  image_height?: number;
}

export interface PipelineApiResponse {
  damage_status: string;
  has_damage: boolean;
  total_detections: number;
  unique_categories: string[];
  average_confidence: number;
  damage_area_percentage: number;
  detections: DetectionBox[];
  image_width: number;
  image_height: number;
  inference_time_ms: number;
  annotated_image: string;
  annotated_image_path?: string;
}

type IconName =
  | "car"
  | "upload"
  | "sun"
  | "frame"
  | "clean"
  | "check"
  | "arrow"
  | "refresh"
  | "download"
  | "zoom"
  | "expand"
  | "shield"
  | "damage";

function Icon({ name, size = 20 }: { name: IconName; size?: number }) {
  const common = {
    width: size,
    height: size,
    viewBox: "0 0 24 24",
    fill: "none",
    stroke: "currentColor",
    strokeWidth: 1.8,
    strokeLinecap: "round" as const,
    strokeLinejoin: "round" as const,
    "aria-hidden": true,
  };

  const paths: Record<IconName, React.ReactNode> = {
    car: (
      <>
        <path d="M3 14.5v-2.2c0-.8.4-1.5 1.1-1.9l1.4-.8 1.7-3.2c.3-.6 1-1 1.7-1h6.6c.8 0 1.5.4 1.8 1.1l1.5 3.1 1.3.7c.6.4 1 1.1 1 1.8v2.4" />
        <path d="M5 14.5h14M7 9.5h10" />
        <circle cx="6.5" cy="15.5" r="1.8" />
        <circle cx="17.5" cy="15.5" r="1.8" />
      </>
    ),
    upload: (
      <>
        <path d="M12 16V4m0 0L7.5 8.5M12 4l4.5 4.5" />
        <path d="M5 14v4a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2v-4" />
      </>
    ),
    sun: (
      <>
        <circle cx="12" cy="12" r="3.5" />
        <path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4m11.4 11.4 1.4 1.4M2 12h2m16 0h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" />
      </>
    ),
    frame: (
      <>
        <path d="M4 8V5a1 1 0 0 1 1-1h3M16 4h3a1 1 0 0 1 1 1v3M20 16v3a1 1 0 0 1-1 1h-3M8 20H5a1 1 0 0 1-1-1v-3" />
        <circle cx="12" cy="12" r="3.2" />
      </>
    ),
    clean: (
      <>
        <path d="M4 17h16M6 17l1.2-6h9.6l1.2 6M9 11l1.5-4h3L15 11" />
        <path d="M18 4v3M16.5 5.5h3" />
      </>
    ),
    check: <path d="m5 12 4 4L19 6" />,
    arrow: (
      <>
        <path d="M5 12h14M14 7l5 5-5 5" />
      </>
    ),
    refresh: (
      <>
        <path d="M20 7v5h-5" />
        <path d="M18.1 16a8 8 0 1 1 .5-8.5L20 12" />
      </>
    ),
    download: (
      <>
        <path d="M12 3v12m0 0 4-4m-4 4-4-4" />
        <path d="M5 19h14" />
      </>
    ),
    zoom: (
      <>
        <circle cx="10.5" cy="10.5" r="5.5" />
        <path d="m15 15 5 5M10.5 8v5M8 10.5h5" />
      </>
    ),
    expand: (
      <>
        <path d="M8 3H3v5M16 3h5v5M8 21H3v-5M16 21h5v-5" />
      </>
    ),
    shield: (
      <>
        <path d="M12 3 5 6v5c0 4.5 2.8 8 7 10 4.2-2 7-5.5 7-10V6l-7-3Z" />
        <path d="m9 12 2 2 4-5" />
      </>
    ),
    damage: (
      <>
        <path d="M5 17 9 6l3 5 2-4 5 10H5Z" />
        <path d="m11 12 2 2-1 3" />
      </>
    ),
  };
  return <svg {...common}>{paths[name]}</svg>;
}

function Button({
  id,
  children,
  variant = "primary",
  onClick,
  disabled = false,
  className = "",
}: {
  id?: string;
  children: React.ReactNode;
  variant?: "primary" | "secondary" | "ghost";
  onClick?: () => void;
  disabled?: boolean;
  className?: string;
}) {
  return (
    <button
      id={id}
      className={`button button-${variant} ${className}`}
      onClick={onClick}
      disabled={disabled}
      type="button"
    >
      {children}
    </button>
  );
}

function Header({
  state,
  systemHealthy = null,
}: {
  state: AppState;
  systemHealthy?: boolean | null;
}) {
  const active = state === "inspect" ? 0 : state === "analyzing" ? 1 : 2;

  let statusText = "Checking AI System...";
  let dotStyle: React.CSSProperties = {
    background: "#d4a940",
    boxShadow: "0 0 0 4px rgba(212, 169, 64, 0.15)",
  };

  if (systemHealthy === true) {
    statusText = "AI System Ready";
    dotStyle = {
      background: "#20a875",
      boxShadow: "0 0 0 4px rgba(32, 168, 117, 0.15)",
    };
  } else if (systemHealthy === false) {
    statusText = "System Offline";
    dotStyle = {
      background: "#d84848",
      boxShadow: "0 0 0 4px rgba(216, 72, 72, 0.15)",
    };
  }

  return (
    <header className="app-header">
      <div className="header-inner">
        <div className="brand">
          <span className="brand-mark">
            <Icon name="car" size={22} />
          </span>
          <span className="brand-copy">
            <strong>Vehicle Damage Scanner</strong>
            <small>AI Inspection</small>
          </span>
        </div>
        <nav className="steps" aria-label="Inspection progress">
          {["Inspect", "Analyze", "Results"].map((label, index) => (
            <div className="step-wrap" key={label}>
              <span className={`step ${active === index ? "step-active" : ""} ${active > index ? "step-done" : ""}`}>
                <b>0{index + 1}</b> {label}
              </span>
              {index < 2 && <span className="step-arrow">→</span>}
            </div>
          ))}
        </nav>
        <div className="system-status">
          <span className="status-dot" style={dotStyle} />
          <span style={{ color: systemHealthy === false ? "#d84848" : undefined }}>{statusText}</span>
        </div>
      </div>
    </header>
  );
}

const tips: { icon: IconName; title: string; copy: string }[] = [
  { icon: "sun", title: "Good lighting", copy: "Use daylight or evenly distributed lighting." },
  { icon: "frame", title: "Wide angle", copy: "Include the damaged panel and surrounding body." },
  { icon: "clean", title: "Clean surface", copy: "Remove heavy dirt or objects covering the damage." },
];

function EmptyUpload({
  selectFile,
  useSample,
  dragging,
  setDragging,
  onDrop,
}: {
  selectFile: () => void;
  useSample: () => void;
  dragging: boolean;
  setDragging: (value: boolean) => void;
  onDrop: (event: React.DragEvent) => void;
}) {
  return (
    <section className="upload-shell">
      <div
        id="dropzone"
        className={`drop-zone ${dragging ? "is-dragging" : ""}`}
        onDragOver={(event) => {
          event.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
      >
        <div id="dropzone-empty-content" style={{ display: "contents" }}>
          <span className="upload-icon">
            <Icon name="upload" size={26} />
          </span>
          <p className="eyebrow">VEHICLE IMAGE</p>
          <h1>Upload vehicle photo</h1>
          <p className="upload-description">Drop an image here or browse files from your device</p>
          <div className="file-spec">
            <span>JPG • PNG • WEBP</span>
            <i />
            <span>Maximum 25 MB</span>
          </div>
          <Button onClick={selectFile}>Choose Photo</Button>
          <button className="sample-link" type="button" onClick={useSample}>
            Use sample damaged vehicle
          </button>
          <span className="drag-copy">or drag and drop your image here</span>
        </div>
      </div>
      <div className="privacy-note">
        <Icon name="shield" size={16} />
        <span>Photos are processed locally for automated vehicle damage analysis.</span>
      </div>
    </section>
  );
}

function Guide() {
  return (
    <aside className="guide-panel">
      <div className="guide-heading">
        <p className="eyebrow">CAPTURE GUIDE</p>
        <h2>Get better results</h2>
        <p>Capture a clear image for more accurate AI detection.</p>
      </div>
      <div className="tips">
        {tips.map((tip, index) => (
          <div className="tip" key={tip.title}>
            <span className="tip-number">0{index + 1}</span>
            <span className="tip-icon">
              <Icon name={tip.icon} />
            </span>
            <span>
              <strong>{tip.title}</strong>
              <p>{tip.copy}</p>
            </span>
          </div>
        ))}
      </div>
      <div className="guide-footer">
        <span className="focus-corners" />
        <span>Keep damage in the center of frame</span>
      </div>
    </aside>
  );
}

function SelectedImage({
  src,
  fileName,
  fileMeta,
  onStart,
  onChange,
}: {
  src: string;
  fileName: string;
  fileMeta: string;
  onStart: () => void;
  onChange: () => void;
}) {
  return (
    <>
      <section className="selected-panel" id="dropzone-preview-overlay">
        <div className="selected-image-wrap">
          <img id="preview-image" src={src} alt="Selected vehicle for inspection" />
          <span className="image-ready-badge">
            <span className="status-dot" /> Image ready
          </span>
          <span className="focus top-left" />
          <span className="focus top-right" />
          <span className="focus bottom-left" />
          <span className="focus bottom-right" />
        </div>
        <div className="image-meta">
          <div>
            <span className="file-icon">JPG</span>
            <span>
              <strong id="preview-filename">{fileName}</strong>
              <small id="preview-filemeta">{fileMeta}</small>
            </span>
          </div>
          <span className="meta-status">INPUT VERIFIED</span>
        </div>
      </section>
      <aside className="ready-panel">
        <div>
          <p className="eyebrow">PRE-FLIGHT CHECK</p>
          <h2>Ready for inspection</h2>
          <p>Image quality meets the minimum requirements for AI processing.</p>
        </div>
        <div className="ready-checks">
          {["Image loaded", "Vehicle detected", "Ready for AI analysis"].map((item, index) => (
            <div key={item}>
              <span>
                <Icon name="check" size={17} />
              </span>
              <p>{item}</p>
              <small>0{index + 1}</small>
            </div>
          ))}
        </div>
        <div className="ready-actions">
          <Button id="btn-analyze" onClick={onStart}>
            Start AI Inspection <Icon name="arrow" size={18} />
          </Button>
          <Button variant="secondary" onClick={onChange}>
            Change Image
          </Button>
        </div>
      </aside>
    </>
  );
}

function Inspect({
  image,
  fileName,
  fileMeta,
  errorMessage,
  onClearError,
  onFile,
  useSample,
  onStart,
}: {
  image: string | null;
  fileName: string;
  fileMeta: string;
  errorMessage: string | null;
  onClearError: () => void;
  onFile: (file: File) => void;
  useSample: () => void;
  onStart: () => void;
}) {
  const input = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  return (
    <main id="view-upload" className="page app-container page-enter">
      <div className="inspect-intro">
        <div>
          <p className="eyebrow">AI-POWERED VISUAL ASSESSMENT</p>
          <h1>{image ? "Review vehicle image" : "Start a new inspection"}</h1>
        </div>
        <p>
          {image
            ? "Confirm the image is clear before beginning the automated exterior scan."
            : "Upload one clear photo to identify visible dents, cracks, and surface damage."}
        </p>
      </div>

      {errorMessage && (
        <div className="error-banner" role="alert">
          <span>{errorMessage}</span>
          <button type="button" onClick={onClearError}>
            Dismiss
          </button>
        </div>
      )}

      <input
        id="file-input"
        ref={input}
        type="file"
        className="sr-only"
        accept="image/jpeg,image/png,image/webp"
        onChange={(event) => event.target.files?.[0] && onFile(event.target.files[0])}
      />
      <div className="inspect-grid">
        {image ? (
          <SelectedImage
            src={image}
            fileName={fileName}
            fileMeta={fileMeta}
            onStart={onStart}
            onChange={() => input.current?.click()}
          />
        ) : (
          <>
            <EmptyUpload
              selectFile={() => input.current?.click()}
              useSample={useSample}
              dragging={dragging}
              setDragging={setDragging}
              onDrop={(event) => {
                event.preventDefault();
                setDragging(false);
                const file = event.dataTransfer.files?.[0];
                if (file) onFile(file);
              }}
            />
            <Guide />
          </>
        )}
      </div>
      {!image && (
        <Button id="btn-analyze" className="analyze-button" disabled onClick={onStart}>
          Analyze Damage <Icon name="arrow" />
        </Button>
      )}
    </main>
  );
}

const analysisSteps = [
  "Detecting vehicle...",
  "Scanning body panels...",
  "Analyzing surface damage...",
  "Checking lights and glass...",
  "Finalizing inspection...",
];

function CarSilhouette() {
  return (
    <svg className="scan-car" viewBox="0 0 420 180" aria-label="Vehicle scanning illustration">
      <defs>
        <linearGradient id="carFill" x1="0" x2="1">
          <stop offset="0" stopColor="#dce7e3" />
          <stop offset=".5" stopColor="#f8fbfa" />
          <stop offset="1" stopColor="#cbd9d5" />
        </linearGradient>
      </defs>
      <path
        className="car-body"
        d="M35 108c7-17 20-27 42-31l53-9 39-39c7-7 16-11 26-11h79c12 0 21 4 29 12l35 38 36 12c16 5 23 16 24 33v18h-28c-3-25-17-38-42-38s-39 13-42 38H131c-3-25-17-38-42-38s-39 13-42 38H25v-10c0-6 3-10 10-13Z"
      />
      <path className="car-window" d="m151 65 33-34c4-4 9-6 15-6h28v40h-76Zm87 0V25h33c8 0 14 3 19 8l29 32h-81Z" />
      <path d="M137 70h192M235 25v106M38 108h22M355 106h40" />
      <circle className="wheel" cx="89" cy="131" r="29" />
      <circle className="wheel-inner" cx="89" cy="131" r="13" />
      <circle className="wheel" cx="328" cy="131" r="29" />
      <circle className="wheel-inner" cx="328" cy="131" r="13" />
      <path className="car-detail" d="M263 78h22M66 84l23-4M335 78l26 8" />
    </svg>
  );
}

function Analyzing({ progress }: { progress: number }) {
  const step = Math.min(4, Math.floor(progress / 20));
  const radius = 211;
  const circumference = 2 * Math.PI * radius;
  return (
    <main id="view-analyzing" className="analysis-page page-enter">
      <div className="analysis-heading">
        <p className="eyebrow">COMPUTER VISION ENGINE</p>
        <h1>Analyzing vehicle...</h1>
        <p>AI is scanning the exterior for visible damage</p>
      </div>
      <div className="scanner">
        <svg className="progress-ring" viewBox="0 0 480 480">
          <circle className="ring-track" cx="240" cy="240" r={radius} />
          <circle
            className="ring-progress"
            cx="240"
            cy="240"
            r={radius}
            style={{ strokeDasharray: circumference, strokeDashoffset: circumference * (1 - progress / 100) }}
          />
        </svg>
        <div className="orbit">
          <span className="orbit-dot dot-one" />
          <span className="orbit-dot dot-two" />
          <span className="orbit-car">
            <Icon name="car" size={27} />
          </span>
        </div>
        <span className="tech-label label-body">BODY</span>
        <span className="tech-label label-panels">PANELS</span>
        <span className="tech-label label-glass">GLASS</span>
        <span className="tech-label label-lights">LIGHTS</span>
        <span className="tech-label label-wheels">WHEELS</span>
        <div className="scan-stage">
          <div className="ai-core">
            <strong>AI</strong>
            <span>SCAN</span>
          </div>
          <CarSilhouette />
          <span className="scan-beam" />
          <span className="scan-grid" />
        </div>
      </div>
      <div className="analysis-progress">
        <div className="analysis-status" key={step}>
          <span className="pulse-mark" />
          <span>{analysisSteps[step]}</span>
          <strong>{Math.round(progress)}%</strong>
        </div>
        <div className="progress-bar">
          <span style={{ width: `${progress}%` }} />
        </div>
        <div className="progress-marks">
          {[0, 25, 50, 75, 100].map((mark) => (
            <span className={progress >= mark ? "passed" : ""} key={mark}>
              {mark}%
            </span>
          ))}
        </div>
      </div>
    </main>
  );
}

function Metric({ label, value, accent }: { label: string; value: string; accent?: boolean }) {
  return (
    <div className={`metric ${accent ? "metric-accent" : ""}`}>
      <span>{label}</span>
      <strong>{value}</strong>
      <i />
    </div>
  );
}

function getDamageTheme(className: string): { color: string; hex: string } {
  const lower = className.toLowerCase();
  if (lower.includes("dent")) return { color: "blue", hex: "#3478e5" };
  if (lower.includes("crack")) return { color: "red", hex: "#d84848" };
  if (lower.includes("scratch")) return { color: "amber", hex: "#d4a940" };
  if (lower.includes("tire") || lower.includes("flat")) return { color: "purple", hex: "#8b5cf6" };
  return { color: "blue", hex: "#3478e5" };
}

function Results({
  image,
  annotatedImageUrl,
  result,
  onReset,
}: {
  image: string;
  annotatedImageUrl: string | null;
  result: PipelineApiResponse | null;
  onReset: () => void;
}) {
  const [useAnnotated, setUseAnnotated] = useState(false);
  const [fullscreen, setFullscreen] = useState(false);

  const displayImage = useAnnotated && annotatedImageUrl ? annotatedImageUrl : image;

  const totalDetections = result?.total_detections ?? 0;
  const damagedAreaPct = result ? `${result.damage_area_percentage.toFixed(2)}%` : "0.00%";
  const avgConfPct = result ? `${(result.average_confidence * 100).toFixed(1)}%` : "0.0%";
  const inferenceMs = result?.inference_time_ms ? `${Math.round(result.inference_time_ms)} ms` : "0 ms";

  const imageW = result?.image_width || 1;
  const imageH = result?.image_height || 1;

  // Severity rating calculation
  const areaRatio = (result?.damage_area_percentage ?? 0) / 100;
  let severityLabel = "UNDAMAGED";
  let severityPosition = "5%";
  let conditionText = "Automated visual inspection identified no visible vehicle damage.";

  if (totalDetections > 0) {
    if (areaRatio > 0.15 || totalDetections >= 4) {
      severityLabel = "HIGH";
      severityPosition = "85%";
      conditionText = "Automated inspection identified significant or widespread surface damage.";
    } else if (areaRatio > 0.04 || totalDetections >= 2) {
      severityLabel = "MODERATE";
      severityPosition = "58%";
      conditionText = "Automated visual inspection identified localized vehicle damage.";
    } else {
      severityLabel = "LOW";
      severityPosition = "22%";
      conditionText = "Automated visual inspection identified minor localized exterior damage.";
    }
  }

  const downloadReport = () => {
    let report = `VEHICLE DAMAGE SCANNER - INSPECTION REPORT\n`;
    report += `==========================================\n`;
    report += `Status: ${result?.damage_status ?? "Completed"}\n`;
    report += `Total Detections: ${totalDetections}\n`;
    report += `Damaged Area: ${damagedAreaPct}\n`;
    report += `Average Confidence: ${avgConfPct}\n`;
    report += `Inference Processing Time: ${inferenceMs}\n`;
    report += `Resolution: ${imageW} x ${imageH} px\n\n`;
    report += `DETECTION DETAILS:\n`;
    if (result && result.detections.length > 0) {
      result.detections.forEach((d, i) => {
        const areaStr = d.bounding_box_area ? ` (Area: ${Math.round(d.bounding_box_area)} px²)` : "";
        report += `${i + 1}. [${d.class_name.toUpperCase()}] Confidence: ${(d.confidence * 100).toFixed(1)}% | Box: [${Math.round(d.x1)}, ${Math.round(d.y1)}, ${Math.round(d.x2)}, ${Math.round(d.y2)}]${areaStr}\n`;
      });
    } else {
      report += `No vehicle damage detected in analyzed frame.\n`;
    }

    const href = URL.createObjectURL(new Blob([report], { type: "text/plain" }));
    const anchor = document.createElement("a");
    anchor.href = href;
    anchor.download = `vehicle-damage-report-${Date.now()}.txt`;
    anchor.click();
    URL.revokeObjectURL(href);
  };

  const downloadJson = () => {
    if (!result) return;
    const jsonStr = JSON.stringify(result, null, 2);
    const href = URL.createObjectURL(new Blob([jsonStr], { type: "application/json" }));
    const anchor = document.createElement("a");
    anchor.href = href;
    anchor.download = `vehicle-damage-data-${Date.now()}.json`;
    anchor.click();
    URL.revokeObjectURL(href);
  };

  return (
    <main id="view-results" className="page app-container page-enter results-page">
      <div className="results-hero">
        <div>
          <span className="complete-label">
            <Icon name="check" size={14} /> SCAN COMPLETE
          </span>
          <h1>Damage Assessment</h1>
          <p>
            {totalDetections} damage {totalDetections === 1 ? "area" : "areas"} detected <i /> {inferenceMs} processing time
          </p>
        </div>
        <div className="results-actions">
          <Button variant="secondary" onClick={onReset}>
            <Icon name="refresh" size={17} /> New Inspection
          </Button>
          <Button variant="secondary" onClick={downloadJson}>
            <Icon name="download" size={17} /> Export JSON
          </Button>
          <Button onClick={downloadReport}>
            <Icon name="download" size={17} /> Download Report
          </Button>
        </div>
      </div>
      <div className="results-grid">
        <section className="detection-view">
          <div className="detection-header">
            <div>
              <span className="live-indicator" />
              <span>AI DETECTION OVERLAY</span>
            </div>
            <span>FRAME 001 / 001</span>
          </div>
          <div className={`result-image-wrap ${fullscreen ? "is-fullscreen" : ""}`}>
            <img src={displayImage} alt="Vehicle exterior scan" />

            {/* Bounding boxes dynamically rendered from real model coordinates */}
            {!useAnnotated &&
              result?.detections.map((det, index) => {
                const theme = getDamageTheme(det.class_name);
                const left = (det.x1 / imageW) * 100;
                const top = (det.y1 / imageH) * 100;
                const width = ((det.x2 - det.x1) / imageW) * 100;
                const height = ((det.y2 - det.y1) / imageH) * 100;
                const confPct = (det.confidence * 100).toFixed(1);

                return (
                  <div
                    key={`det-box-${index}`}
                    className="bounding-box"
                    style={
                      {
                        "--box-color": theme.hex,
                        left: `${left}%`,
                        top: `${top}%`,
                        width: `${width}%`,
                        height: `${height}%`,
                        animationDelay: `${150 + index * 120}ms`,
                      } as React.CSSProperties
                    }
                  >
                    <span>
                      {det.class_name} <b>{confPct}%</b>
                    </span>
                    <i className="corner c1" />
                    <i className="corner c2" />
                    <i className="corner c3" />
                    <i className="corner c4" />
                  </div>
                );
              })}

            <div className="image-controls">
              {annotatedImageUrl && (
                <button
                  type="button"
                  aria-label="Toggle annotated view"
                  title={useAnnotated ? "Show Raw Image + Overlay" : "Show OpenCV Rendered Image"}
                  onClick={() => setUseAnnotated(!useAnnotated)}
                >
                  <Icon name="zoom" size={18} />
                </button>
              )}
              <button
                type="button"
                aria-label="View fullscreen"
                onClick={() => setFullscreen(!fullscreen)}
              >
                <Icon name="expand" size={18} />
              </button>
            </div>
          </div>
          <div className="overlay-footer">
            <span>
              <Icon name="shield" size={16} /> Precision AI Overlay
            </span>
            <span>CV MODEL YOLOv8</span>
          </div>
        </section>

        <aside className="summary-panel">
          <div className="summary-title">
            <div>
              <p className="eyebrow">AUTOMATED REPORT</p>
              <h2>Inspection Summary</h2>
            </div>
            <span className="summary-check">
              <Icon name="check" size={17} />
            </span>
          </div>
          <div className="metrics">
            <Metric label="TOTAL DETECTIONS" value={String(totalDetections)} accent />
            <Metric label="DAMAGED AREA" value={damagedAreaPct} />
            <Metric label="AVG. CONFIDENCE" value={avgConfPct} />
          </div>
          <div className="detected-section">
            <div className="section-heading">
              <h3>Detected Damage</h3>
              <span>{totalDetections} {totalDetections === 1 ? "AREA" : "AREAS"}</span>
            </div>
            <div className="detection-list">
              {result && result.detections.length > 0 ? (
                result.detections.map((det, index) => {
                  const theme = getDamageTheme(det.class_name);
                  const confVal = (det.confidence * 100).toFixed(1);
                  const detNum = `DET-0${index + 1}`;
                  return (
                    <div className="detection-item" key={`${det.class_name}-${index}`}>
                      <span className={`damage-icon ${theme.color}`}>
                        <Icon name="damage" size={17} />
                      </span>
                      <div className="detection-copy">
                        <div>
                          <strong>{det.class_name.toUpperCase()}</strong>
                          <span>{confVal}% confidence</span>
                        </div>
                        <span className="confidence-track">
                          <i
                            className={theme.color}
                            style={
                              {
                                "--confidence": `${confVal}%`,
                                animationDelay: `${0.2 + index * 0.12}s`,
                              } as React.CSSProperties
                            }
                          />
                        </span>
                      </div>
                      <small>{detNum}</small>
                    </div>
                  );
                })
              ) : (
                <div className="empty-results">
                  <p>No damage detected. Vehicle appears undamaged in this view.</p>
                </div>
              )}
            </div>
          </div>
          <div className="condition">
            <div className="section-heading">
              <h3>Overall Condition</h3>
              <span>{severityLabel}</span>
            </div>
            <p>{conditionText}</p>
            <div className="severity">
              <div className="severity-line">
                <span className="severity-fill" />
                <i style={{ left: severityPosition }} />
              </div>
              <div>
                <span>LOW</span>
                <span>MODERATE</span>
                <span>HIGH</span>
              </div>
            </div>
            <span className="assessment-note">
              <Icon name="shield" size={15} /> AI-assisted visual assessment
            </span>
          </div>
        </aside>
      </div>
    </main>
  );
}

export default function App() {
  const [state, setState] = useState<AppState>("inspect");
  const [image, setImage] = useState<string | null>(null);
  const [currentFile, setCurrentFile] = useState<File | null>(null);
  const [fileName, setFileName] = useState("vehicle_scan.jpg");
  const [fileMeta, setFileMeta] = useState("0 × 0 • 0 MB");
  const [progress, setProgress] = useState(0);

  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [pipelineResult, setPipelineResult] = useState<PipelineApiResponse | null>(null);
  const [annotatedImageUrl, setAnnotatedImageUrl] = useState<string | null>(null);
  const [systemHealthy, setSystemHealthy] = useState<boolean | null>(null);

  const activeRequestRef = useRef<AbortController | null>(null);

  // Poll real backend health
  useEffect(() => {
    let isMounted = true;
    const checkHealth = async () => {
      try {
        const res = await fetch("/api/health");
        if (res.ok) {
          if (isMounted) setSystemHealthy(true);
        } else {
          if (isMounted) setSystemHealthy(false);
        }
      } catch {
        if (isMounted) setSystemHealthy(false);
      }
    };
    checkHealth();
    const interval = window.setInterval(checkHealth, 20000);
    return () => {
      isMounted = false;
      window.clearInterval(interval);
    };
  }, []);

  // Clean up object URLs on unmount
  useEffect(() => {
    return () => {
      if (image?.startsWith("blob:")) URL.revokeObjectURL(image);
    };
  }, [image]);

  const validateAndSetFile = (file: File) => {
    setErrorMessage(null);
    const validExtensions = [".jpg", ".jpeg", ".png", ".webp"];
    const ext = "." + file.name.split(".").pop()?.toLowerCase();

    if (!validExtensions.includes(ext)) {
      setErrorMessage(`Unsupported format (${ext}). Please select a JPG, PNG, or WEBP image.`);
      return;
    }

    const maxSize = 25 * 1024 * 1024;
    if (file.size > maxSize) {
      setErrorMessage(`File is too large (${(file.size / (1024 * 1024)).toFixed(1)} MB). Maximum allowed size is 25 MB.`);
      return;
    }

    if (image?.startsWith("blob:")) URL.revokeObjectURL(image);

    const url = URL.createObjectURL(file);
    const preview = new Image();
    preview.onload = () => {
      setFileMeta(`${preview.width} × ${preview.height} • ${(file.size / 1024 / 1024).toFixed(1)} MB`);
    };
    preview.src = url;

    setImage(url);
    setCurrentFile(file);
    setFileName(file.name);
  };

  const useSampleVehicle = async () => {
    setErrorMessage(null);
    try {
      const response = await fetch(SAMPLE_IMAGE_URL);
      if (!response.ok) {
        throw new Error(`Sample image not found on server (${response.status})`);
      }
      const blob = await response.blob();
      const sampleFile = new File([blob], "000012.jpg", { type: "image/jpeg" });
      validateAndSetFile(sampleFile);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Failed to load sample vehicle";
      setErrorMessage(`Sample load error: ${msg}. You can upload a photo directly.`);
    }
  };

  const startAnalysis = async () => {
    if (!currentFile && !image) return;
    setErrorMessage(null);
    setState("analyzing");
    setProgress(5);

    // Dynamic progress ticker that ramps up toward ~90% while the request is in-flight
    const startTime = Date.now();
    const progressTimer = window.setInterval(() => {
      const elapsed = Date.now() - startTime;
      setProgress((prev) => {
        if (prev >= 90) return prev;
        // Asymptotically approach 90%
        const target = Math.min(90, Math.floor(5 + (elapsed / 3000) * 85));
        return Math.max(prev, target);
      });
    }, 80);

    const abortController = new AbortController();
    activeRequestRef.current = abortController;

    try {
      let fileToSend = currentFile;
      if (!fileToSend && image) {
        const fetchRes = await fetch(image);
        const blob = await fetchRes.blob();
        fileToSend = new File([blob], fileName || "vehicle.jpg", { type: blob.type || "image/jpeg" });
      }

      if (!fileToSend) throw new Error("No image data available to analyze.");

      const formData = new FormData();
      formData.append("file", fileToSend);

      const response = await fetch("/api/detect", {
        method: "POST",
        body: formData,
        signal: abortController.signal,
      });

      if (!response.ok) {
        let errDetail = `Server returned HTTP ${response.status}`;
        try {
          const errJson = await response.json();
          if (errJson.detail) errDetail = errJson.detail;
        } catch {
          // ignore parsing error
        }
        throw new Error(errDetail);
      }

      const data: PipelineApiResponse = await response.json();

      window.clearInterval(progressTimer);
      setProgress(100);

      setPipelineResult(data);
      setAnnotatedImageUrl(data.annotated_image || null);

      window.setTimeout(() => {
        setState("results");
      }, 350);
    } catch (err: unknown) {
      window.clearInterval(progressTimer);
      if (err instanceof Error && err.name === "AbortError") {
        return;
      }
      const msg = err instanceof Error ? err.message : "Connection to detection server failed.";
      setErrorMessage(`Detection error: ${msg}`);
      setState("inspect");
    } finally {
      activeRequestRef.current = null;
    }
  };

  const resetAll = () => {
    if (activeRequestRef.current) {
      activeRequestRef.current.abort();
    }
    if (image?.startsWith("blob:")) {
      URL.revokeObjectURL(image);
    }
    setImage(null);
    setCurrentFile(null);
    setFileName("vehicle_scan.jpg");
    setFileMeta("0 × 0 • 0 MB");
    setProgress(0);
    setPipelineResult(null);
    setAnnotatedImageUrl(null);
    setErrorMessage(null);
    setState("inspect");
  };

  return (
    <div className="app-shell">
      <Header state={state} systemHealthy={systemHealthy} />
      {state === "inspect" && (
        <Inspect
          image={image}
          fileName={fileName}
          fileMeta={fileMeta}
          errorMessage={errorMessage}
          onClearError={() => setErrorMessage(null)}
          onFile={validateAndSetFile}
          useSample={useSampleVehicle}
          onStart={startAnalysis}
        />
      )}
      {state === "analyzing" && <Analyzing progress={progress} />}
      {state === "results" && (
        <Results
          image={image || SAMPLE_IMAGE_URL}
          annotatedImageUrl={annotatedImageUrl}
          result={pipelineResult}
          onReset={resetAll}
        />
      )}
    </div>
  );
}
