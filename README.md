# 智慧型合金壓鑄模具瑕疵檢測系統  
**Intelligent Aluminum Alloy Die-Casting Mold Defect Detection System**

## 專案簡介 | Project Overview

本系統旨在透過高解析度工業相機與人工智慧模型，自動化檢測合金壓鑄模具之瑕疵。應用範圍包括金屬加工、自動光學檢測、品質控制等領域，提升工業自動化與智慧製造效率。

The system aims to detect defects in aluminum alloy die-casting molds using high-resolution industrial imaging and AI-based models, automating the quality inspection process in metalworking and manufacturing.

---

## 系統功能需求 | Functional Requirements

| 編號 | 功能需求描述 |
|------|---------------|
| AOI-F-001 | 系統可透過 **Hikrobot MVS 工業相機**自動擷取平台上物件的影像。 |
| AOI-F-002 | 系統應使用 AI 模組（如 YOLO）辨識潛在缺陷的區域。 |
| AOI-F-003 | 系統應具備元件類型分類功能（例如辨識不同模具類型）。 |
| AOI-F-004 | 系統應能判斷是否存在缺陷，並進一步分類缺陷類型（如刮痕、凹陷等）。 |
| AOI-F-005 | 分析結果應即時顯示於使用者介面中，供使用者參考。 |
| AOI-F-006 | 所有檢測影像與資料應自動儲存並記錄時間與缺陷類型，以利後續查詢與追蹤。 |
| AOI-F-007 | 系統應提供依據日期與分類檢索並查看歷史資料的功能。 |

---

## 系統非功能需求 | Non-Functional Requirements

| 編號 | 非功能需求描述 |
|------|----------------|
| AOI-NF-001 | 系統應可穩定運作長時間（至少連續運行 24 小時以上無需重啟）。 |
| AOI-NF-002 | 使用者介面應支援中英文語言切換。 |
| AOI-NF-003 | 所有影像與分析結果應自動儲存至本地端資料庫。 |
| AOI-NF-004 | 單張影像處理時間（含缺陷判斷）不得超過 2 秒。 |
| AOI-NF-005 | 系統應支援 **Windows 10+** 與 **Ubuntu 20.04+** 作業系統。 |
| AOI-NF-006 | 使用者介面應簡潔直覺，並可即時顯示處理結果。 |

---

## 使用案例與腳本 | Use Cases & Scenarios

### UC001：系統啟動與介面操作

**使用者操作流程：**
1. 載入 AOI 操作介面  
2. 啟動程式  
3. 選擇操作模式  
4. 檢查相機與光源連線狀態

---

### UC002：影像擷取與分析啟動

**使用者操作流程：**
1. 使用者將產品放置於金屬平台上  
2. 點選「掃描」按鈕  
3. 相機拍攝產品圖像  
4. 系統將影像傳送至神經網路模型進行分析

---

### UC003：顯示與儲存分析結果

**使用者操作流程：**
1. 系統即時顯示分析結果於 UI  
2. 使用者確認視覺化結果  
3. 系統將影像與缺陷資訊儲存至資料庫  
4. 使用者可選擇匯出或列印報告


## Usage 使用說明

1. 複製此儲存庫（Clone the repo）
```bash
git clone https://github.com/KasymovD/aluminum-defect-detector.git
```
2. 安裝相依套件（Install dependencies）

```bash
pip install -r requirements.txt
```

3. 執行檢測腳本（Run the detection script）

```bash
python main.py
```

## Author