# AI Chatbot Prompt for Final Presentation Slides

Copy everything below this line into ChatGPT, Claude, or Google Gemini to generate your presentation slides.

---

Create a professional Google Slides presentation for an AI4ALL Ignite Final Presentation. This must score 30/30 on the rubric (6 criteria worth 5 points each). Keep it under 10 minutes (12-15 slides total).

## PROJECT INFORMATION

**Title:** Water Quality Prediction Using Machine Learning
**Team Members:**
- Sean Esla - App & Infrastructure Lead
- Joseann Boneo - Data & Modeling Lead
- Lademi Aromolaran - Analysis & Communication Lead

**GitHub Repository:** https://github.com/seanesla/group11C

**Research Question:** "How can we use real-world water quality data to support awareness about drinking-water safety, while being honest about uncertainty and bias?"

## WHAT THE APPLICATION DOES (Step by Step)

1. User enters a US ZIP code (e.g., 20001 for Washington DC)
2. App converts ZIP code to latitude/longitude using pgeocode library
3. App queries two government APIs within a configurable radius:
   - USGS National Water Information System (NWIS)
   - Water Quality Portal (WQP) - aggregates data from USGS, EPA, and state agencies
4. App retrieves measurements for 6 water quality parameters:
   - pH (acidity/alkalinity, optimal range 6.5-8.5)
   - Dissolved Oxygen (DO) in mg/L (fish need >5 mg/L)
   - Temperature in Celsius
   - Turbidity in NTU (cloudiness)
   - Nitrate in mg/L as N (fertilizer runoff indicator)
   - Specific Conductance in µS/cm (dissolved minerals)
5. App aggregates measurements into daily values
6. App calculates Water Quality Index (WQI) score 0-100 based on NSF WQI methodology (using 6 of the original 9 NSF parameters, with conductance substituting for total solids):
   - 90-100: Excellent (safe for all uses)
   - 70-89: Good (safe for most uses)
   - 50-69: Fair (limited uses)
   - 25-49: Poor (degraded)
   - 0-24: Very Poor (severely degraded)
7. Machine Learning models predict:
   - Predicted WQI score with confidence interval
   - SAFE/UNSAFE classification (derived from predicted WQI: SAFE if WQI ≥ 70)
   - Classifier probability scores for confidence estimation
8. SHAP (SHapley Additive exPlanations) values show WHY the model made its prediction

**Important Implementation Note:** In the deployed app, the SAFE/UNSAFE label is derived from the regressor's predicted WQI (SAFE if WQI ≥ 70), not directly from the classifier output. The classifier provides probability scores used for confidence estimation.

## ALGORITHM DETAILS

**Algorithm Name:** Random Forest (ensemble of decision trees)
**Type:** Supervised Learning
**Two Models:**

### Model 1: Random Forest Classifier
- **Task:** Binary classification (SAFE vs UNSAFE)
- **Label Definition:** SAFE = WQI >= 70, UNSAFE = WQI < 70
- **Output:** Probability scores for confidence estimation

### Model 2: Random Forest Regressor
- **Task:** Continuous regression
- **Output:** WQI score 0-100 (e.g., "Predicted WQI: 85.3")
- **Note:** The regressor's output determines the app's SAFE/UNSAFE label (SAFE if predicted WQI ≥ 70)

**Input Features (18 total):**
1. year - Calendar year of measurement
2. ph - pH level (0-14 scale)
3. dissolved_oxygen - DO in mg/L
4. temperature - Water temperature in Celsius
5. nitrate - Nitrate concentration in mg/L as N
6. conductance - Specific conductance in µS/cm
7. years_since_1991 - Temporal feature for trend detection
8. decade - Categorical (1990s, 2000s, 2010s)
9. is_1990s - Binary flag
10. is_2000s - Binary flag
11. is_2010s - Binary flag
12. ph_deviation_from_7 - Distance from neutral pH
13. do_temp_ratio - Dissolved oxygen / temperature ratio
14. conductance_low - Binary: conductance < 200 µS/cm
15. conductance_medium - Binary: conductance 200-800 µS/cm
16. conductance_high - Binary: conductance > 800 µS/cm
17. pollution_stress - Interaction: high nitrate × high conductance
18. temp_stress - Interaction: extreme temperature effects

**Training Configuration:**
- 5-fold stratified cross-validation
- GridSearchCV for hyperparameter optimization
- class_weight='balanced' to improve detection of minority class
- 60% training / 20% validation / 20% test split

**Why Random Forest was chosen:**
- PROS:
  - Handles non-linear relationships between water parameters
  - Resistant to overfitting through ensemble averaging
  - Provides built-in feature importance scores
  - Works well with class weights for imbalanced data
  - No need for feature scaling (tree-based)
- CONS:
  - "Black box" - hard to interpret individual decisions (mitigated with SHAP)
  - Requires more memory than single decision tree
  - Slower inference than linear models (acceptable for this use case)

## EXACT MODEL PERFORMANCE METRICS

### Classifier Metrics (test set, n=577):
| Metric | Value | 95% CI |
|--------|-------|--------|
| Accuracy | 98.27% | [97.05%, 99.31%] |
| Precision | 98.17% | [96.53%, 99.40%] |
| Recall | 98.77% | [97.53%, 99.70%] |
| F1 Score | 98.47% | [97.43%, 99.38%] |
| ROC-AUC | 99.82% | [99.52%, 99.98%] |

Confusion Matrix:
- True Negatives: 246 (correctly identified UNSAFE)
- False Positives: 6 (wrongly said SAFE when UNSAFE)
- False Negatives: 4 (wrongly said UNSAFE when SAFE)
- True Positives: 321 (correctly identified SAFE)

Calibration Metrics:
- Brier Score: 0.0141 (measures probability accuracy, lower is better)
- Expected Calibration Error (ECE): 0.0313 (predicted probabilities match actual outcomes)

### Regressor Metrics (test set, n=577):
| Metric | Value | 95% CI |
|--------|-------|--------|
| R² Score | 0.9676 | [0.9362, 0.9890] |
| MAE | 1.67 WQI points | [1.32, 2.09] |
| RMSE | 5.09 WQI points | [2.95, 7.27] |
| Explained Variance | 96.77% | - |

Interpretation: The model explains 96.76% of variance in WQI scores, with typical prediction error of ±1.67 points.

## PROJECT EVOLUTION (How it changed over the program)

**Phase 1 - Weeks 1-3:** Basic WQI Calculator
- Started with simple NSF-style Water Quality Index formula
- Manually input water parameters, output single score

**Phase 2 - Weeks 4-7:** Added Machine Learning
- Trained Random Forest classifier and regressor on Kaggle dataset
- Integrated ML predictions into Streamlit web app
- Added SHAP explanations for model transparency

**Phase 3 - Weeks 8-10:** Critical Discovery - Lead Detection Failure
- Tested model on simulated Flint, Michigan water crisis scenarios
- DISCOVERED: 100% false negative rate on lead-contaminated water
- 150 ppb lead (10× EPA action level) received WQI score of 100.0 "Excellent"
- Root cause: NSF-style WQI methodology does NOT include lead as a parameter

**Phase 4 - Weeks 11-12:** Environmental Justice Analysis
- Documented bias implications for Flint and Jackson communities
- Added explicit warnings about lead detection limitations
- Created comprehensive Environmental Justice Analysis document (1,057 lines)

**Major Challenge Faced:** Moderate class imbalance in training data
- Kaggle training data was approximately 56% SAFE, 44% UNSAFE
- Used class_weight='balanced' parameter in Random Forest to ensure model learns both classes effectively
- This automatically adjusts weights inversely proportional to class frequencies
- Result: Model achieves high recall for both SAFE and UNSAFE classes

**Note:** A separate experiment with a small US-only dataset (128 samples) showed extreme imbalance (98.8% SAFE / 1.2% UNSAFE), demonstrating how real-world deployment data can differ significantly from training data.

## ENVIRONMENTAL JUSTICE ANALYSIS (Critical for Essential Question)

### The Flint, Michigan Water Crisis (2014-2019)

**Demographics (2014 Census estimates):**
- Population: ~99,000
- Black/African American: ~54%
- Below poverty line: ~40%
- Median household income: ~$25,000 (vs. Michigan average ~$49,000)

**What Happened:**
- April 2014: City switched water source from Detroit system to Flint River to save money
- Officials failed to add corrosion control treatment (orthophosphate)
- Corrosive water caused lead to leach from aging pipes into drinking water
- September 2015: Virginia Tech researchers confirmed lead contamination
  - ~40% of tested homes had elevated lead levels (>5 ppb)
  - ~25% exceeded the EPA action level of 15 ppb
  - One home reached 13,200 ppb lead (880× EPA limit)
- Approximately 8,600 children under age 6 were living in Flint during the crisis and should be considered potentially exposed
- Lead causes permanent neurological damage; several IQ points can be lost per 10 µg/dL increase in blood lead level; no safe threshold has been identified

### The Jackson, Mississippi Water Crisis (2021-2023)

**Demographics (2020 Census):**
- Population: ~154,000
- Black/African American: ~80%
- Below poverty line: ~25%
- Median household income: ~$40,000

**What Happened:**
- 100-year-old water infrastructure chronically underfunded
- August 2022: Flooding overwhelmed treatment plant, ~150,000 residents lost running water
- Documented lead levels in the range of 18-30+ ppb (1.2-2× EPA action level)
- Mayor estimated $2 billion needed to fully repair system

### Our System's Failure on Lead-Contaminated Water

We tested 6 contaminated water scenarios based on Flint/Jackson conditions:

| Scenario | Lead Level | Reality | WQI Calculator | ML Classifier | ML Regressor |
|----------|-----------|---------|----------------|---------------|--------------|
| Flint High Lead | 100 ppb (6.7× EPA) | UNSAFE | 93.57 "Excellent" | SAFE (68%) | 92.89 "Excellent" |
| Flint Extreme | 150 ppb (10× EPA) | UNSAFE | 100.0 "Excellent" | SAFE (65%) | 92.89 "Excellent" |
| Flint Moderate | 25 ppb (1.7× EPA) | UNSAFE | 87.06 "Good" | SAFE (58%) | 86.11 "Good" |
| Jackson Treatment Failure | 35 ppb (2.3× EPA) | UNSAFE | 85.95 "Good" | SAFE (54%) | 90.60 "Excellent" |
| Jackson Normal Appearance | 20 ppb (1.3× EPA) | UNSAFE | 91.03 "Excellent" | SAFE (61%) | 93.23 "Excellent" |
| Jackson Chronic | 18 ppb (1.2× EPA) | UNSAFE | 86.27 "Good" | SAFE (53%) | 90.60 "Excellent" |

**Result: 100% FALSE NEGATIVE RATE across all 3 system components on all 6 scenarios**

**Root Cause:** The NSF-style Water Quality Index was designed for surface water ecosystem health, NOT drinking water safety. It measures: pH, dissolved oxygen, temperature, turbidity, nitrate, conductance. It does NOT measure: lead, arsenic, mercury, bacteria (E. coli), PFAS, pesticides.

**Why ML models also failed:** Models were trained on labels derived from WQI (is_safe = WQI >= 70). If WQI can't detect lead, the labels don't account for lead, so models can't learn to detect lead. The models inherited the bias from their training labels.

## ESSENTIAL QUESTION ANSWER

**Question:** "How can the process of creating AI/ML solutions amplify or mitigate bias in the case our group is using?"

### How Our AI AMPLIFIES Bias:

1. **Label Inheritance:** ML models trained on WQI-derived labels inherit WQI's fundamental blind spot to lead contamination. The model learns exactly what we teach it - and we taught it that lead doesn't matter.

2. **Geographic Training Data Bias:** Models trained on European water quality data (1991-2017) don't represent US water conditions, especially infrastructure decay patterns that cause lead contamination in environmental justice communities.

3. **False Confidence Effect:** When AI provides a prediction with a confidence percentage, users may trust it more than a simple formula. A Flint resident seeing "Water Quality: Excellent" could delay protective actions like using bottled water or testing for lead.

4. **Disproportionate Harm to EJ Communities:** Flint (~54% Black, ~40% poverty) and Jackson (~80% Black, ~25% poverty) are exactly the communities most affected by aging infrastructure that causes lead contamination - the exact scenario our model cannot detect.

### How Our AI MITIGATES Bias (Responsible AI practices):

1. **Transparency through SHAP:** We show users exactly WHY the model made its prediction, with per-feature contribution values. Users can see which parameters drove the decision.

2. **Documented Limitations:** We explicitly state what the system CANNOT detect (lead, heavy metals, bacteria, PFAS, pesticides) rather than hiding limitations.

3. **Uncertainty Quantification:** We provide 95% confidence intervals for all metrics, Brier scores for probability calibration, and residual analysis - users know how confident to be.

4. **This Project Itself:** By discovering the 100% false negative rate and documenting it rather than hiding it, we demonstrate responsible AI development. Acknowledging failure is itself a form of bias mitigation.

## POSITIVE AND NEGATIVE IMPACTS

### Positive Impacts:
- Makes government water quality data accessible to non-experts
- Translates complex parameters into understandable 0-100 score
- SHAP explanations promote AI literacy and transparency
- Highlights data coverage gaps in environmental justice communities
- Educational tool for understanding water quality science
- Open source code enables others to learn and build

### Negative Impacts:
- Cannot detect lead, heavy metals, bacteria, PFAS, pesticides - the most dangerous drinking water contaminants
- False sense of security: "Excellent" rating for toxic water could delay protective actions
- Disproportionate risk to environmental justice communities who face lead contamination
- European training data doesn't represent US conditions
- Users may not read disclaimers and trust AI predictions blindly
- No replacement for certified laboratory testing

## DATA SOURCES

### Training Data:
- **Source:** Kaggle Water Quality Dataset
- **Samples:** ~2,900 water quality measurements (after processing)
- **Time Period:** 1991-2017
- **Geography:** Primarily European monitoring sites (France, UK, Spain dominant)
- **Parameters:** pH, dissolved oxygen, temperature, nitrate, conductance (turbidity was missing/NaN in source data)
- **Labels:** WQI score calculated from parameters, is_safe derived from WQI >= 70
- **Class Distribution:** Approximately 56% SAFE / 44% UNSAFE

### Live US Data (Real-time):
- **USGS National Water Information System (NWIS)**
  - Federal agency monitoring sites across US
  - Real-time and historical water quality data
  - API: https://waterservices.usgs.gov/

- **Water Quality Portal (WQP)**
  - Aggregates data from USGS, EPA, and state agencies
  - Most comprehensive US water quality database
  - API: https://www.waterqualitydata.us/

## CITATIONS (Include all 8 on citations slide)

1. **Edwards, M. et al. (2015).** Virginia Tech Flint Water Study. http://flintwaterstudy.org/

2. **U.S. Environmental Protection Agency (2016-2019).** EPA's Response to the Flint Drinking Water Crisis. https://www.epa.gov/flint

3. **Hanna-Attisha, M. et al. (2016).** "Elevated Blood Lead Levels in Children Associated With the Flint Drinking Water Crisis." American Journal of Public Health, 106(2), 283-290.

4. **Centers for Disease Control and Prevention (2016).** "Blood Lead Levels in Children Aged <6 Years - Flint, Michigan, 2013-2016." MMWR Morb Mortal Wkly Rep, 65:650-654.

5. **ProPublica (2023).** "Jackson, Mississippi Residents Don't Trust Their Water - And for Good Reason." https://www.propublica.org/article/jackson-mississippi-water-crisis

6. **U.S. Environmental Protection Agency (2022).** "Biden-Harris Administration Invests $115 million in Funding to Respond to the Drinking Water Emergency in Jackson, Mississippi."

7. **World Health Organization (2023).** "Lead poisoning and health." https://www.who.int/news-room/fact-sheets/detail/lead-poisoning-and-health

8. **Brown, R.M. et al. (1970).** "A Water Quality Index - Do We Dare?" Water & Sewage Works, 117(10), 339-343.

**Data Sources to cite:**
- Kaggle Water Pollution Dataset (training data)
- USGS National Water Information System (live US data)
- Water Quality Portal - USGS, EPA, and state agencies (live US data)

## NEXT STEPS (Must be measurable and specific)

### Individual Next Steps:
- **Sean Esla:** Apply to 3+ ML/data science internships by Spring 2025; complete advanced Python certification; contribute to 2+ open source projects
- **Joseann Boneo:** Enroll in data science courses (Coursera/edX) by January 2025; expand portfolio with 2 additional ML projects; pursue data analytics roles
- **Lademi Aromolaran:** Apply environmental justice research skills to policy internships; write op-ed about AI bias in environmental monitoring; pursue environmental science graduate programs

### Project Next Steps:
- Add lead data fetching from Water Quality Portal (lead characteristic is available in WQP database)
- Implement infrastructure risk indicators (home construction year, pH corrosion risk, known contamination areas)
- Partner with local water utilities for real-time monitoring data
- Develop lead-specific warning system that activates for high-risk ZIP codes (Flint 48502, Jackson 392xx)
- Create community outreach materials in partnership with environmental justice organizations

## SLIDE-BY-SLIDE CONTENT

### Slide 1: Title
**Title:** Water Quality Prediction Using Machine Learning
**Subtitle:** Group 11C | AI4ALL Ignite Final Presentation
**Team:** Sean Esla, Joseann Boneo, Lademi Aromolaran
**Date:** [Insert presentation date]
**Visual:** Include AI4ALL logo, water droplet or wave graphic
**Speaker Notes:** "Good morning/afternoon everyone. We're Group 11C, and we built a machine learning system to predict water quality from government monitoring data. I'm [name], and I'll be presenting with [teammates]."

### Slide 2: The Problem
**Title:** Why Water Quality Prediction Matters
**Content:**
- 2+ million Americans lack running water or basic plumbing (DigDeep/US Water Alliance)
- Flint, MI (2014): ~8,600 children potentially exposed to lead contamination
- Jackson, MS (2022): ~150,000 residents lost running water
- Government data exists but is hard to access and interpret
**Research Question:** "How can we use real-world water quality data to support awareness about drinking-water safety, while being honest about uncertainty and bias?"
**Visual:** Photo of Flint water crisis OR map of US water monitoring gaps
**Speaker Notes:** "These are real crises affecting real communities. More than 2 million Americans lack running water or basic indoor plumbing, according to DigDeep and the US Water Alliance. Our research question asks not just how to predict water quality, but how to do it honestly - acknowledging what our models can and cannot detect."

### Slide 3: Project Overview
**Title:** How Our System Works
**Content:** Pipeline diagram:
ZIP Code → Geocoding (pgeocode) → API Queries (USGS + WQP) → Daily Aggregation → WQI Calculation (0-100) → ML Prediction → SHAP Explanation
**Visual:** Flowchart or architecture diagram
**Speaker Notes:** "Users enter a ZIP code, we convert it to coordinates, query government APIs for water quality measurements, calculate a Water Quality Index score based on NSF methodology, then use machine learning to predict water quality with explainable results."

### Slide 4: Project Evolution
**Title:** How Our Project Evolved
**Content:**
- **Weeks 1-3:** Basic WQI calculator (input parameters, output score)
- **Weeks 4-7:** Added ML models with SHAP explanations
- **Weeks 8-10:** CRITICAL DISCOVERY - 100% false negative on lead
- **Weeks 11-12:** Environmental Justice Analysis
**Challenge:** Training an effective model with class imbalance (~56% SAFE / 44% UNSAFE)
**Solution:** class_weight='balanced' in Random Forest
**Visual:** Timeline graphic
**Speaker Notes:** "Our project evolved significantly. The biggest turning point was discovering that our model gives 'Excellent' ratings to lead-contaminated water. This led us to create a comprehensive environmental justice analysis. We also addressed class imbalance in our training data using balanced class weights."

### Slide 5: Data Sources
**Title:** Our Data
**Training Data:**
- Kaggle Water Quality Dataset
- ~2,900 samples from 1991-2017
- European monitoring sites (France, UK, Spain)
**Live US Data:**
- USGS National Water Information System
- Water Quality Portal (USGS + EPA + state agencies)
**6 Parameters:** pH, Dissolved Oxygen, Temperature, Turbidity, Nitrate, Conductance
**Visual:** Database/API icons, parameter list
**Speaker Notes:** "We trained on a Kaggle dataset with about 2,900 samples from European monitoring sites, then deployed the model to work with live US data from government APIs. The model uses 6 water quality parameters."

### Slide 6: The Algorithm
**Title:** Random Forest: Our Machine Learning Algorithm
**Content:**
- **Type:** Supervised Learning (ensemble of decision trees)
- **Classifier:** Provides probability scores for confidence
- **Regressor:** WQI Score 0-100 (determines SAFE/UNSAFE label)
**Why Random Forest:**
- ✓ Handles non-linear relationships
- ✓ Resistant to overfitting
- ✓ Provides feature importance
- ✓ Works with class weights
- ✗ "Black box" (mitigated with SHAP)
**Visual:** Decision tree ensemble diagram
**Speaker Notes:** "Random Forest is an ensemble method that combines many decision trees. We chose it because it handles our data well and provides feature importance scores for interpretability. The regressor predicts WQI scores, and we derive the SAFE/UNSAFE label from whether the predicted WQI is at least 70."

### Slide 7: Model Inputs & Outputs
**Title:** What Goes In, What Comes Out
**Inputs (18 features):**
- Raw: pH, dissolved oxygen, temperature, nitrate, conductance
- Temporal: year, decade, is_1990s/2000s/2010s
- Derived: pH deviation, DO/temp ratio, conductance bins, stress interactions
**Regressor Output:** WQI 0-100 + category (e.g., "85.3 - Good")
**SAFE/UNSAFE:** Derived from predicted WQI (SAFE if WQI ≥ 70)
**Visual:** Input/Output diagram
**Speaker Notes:** "We engineered 18 features from the raw water parameters. The regressor predicts a continuous WQI score, which determines the safety classification - SAFE if the predicted WQI is 70 or above."

### Slide 8: Model Evaluation [DATA VISUALIZATION #1]
**Title:** Model Performance
**Classifier Results:**
| Metric | Score | 95% CI |
|--------|-------|--------|
| Accuracy | 98.27% | [97.05%, 99.31%] |
| F1 Score | 98.47% | [97.43%, 99.38%] |
| ROC-AUC | 99.82% | [99.52%, 99.98%] |
**Regressor Results:**
| Metric | Score | 95% CI |
|--------|-------|--------|
| R² Score | 96.76% | [93.62%, 98.90%] |
| MAE | 1.67 pts | [1.32, 2.09] |
**Visual:** Bar chart of metrics with confidence interval error bars
**Speaker Notes:** "Our classifier achieves 98% accuracy with tight confidence intervals. The regressor explains nearly 97% of variance in WQI scores with typical error of just 1.7 points. We include 95% confidence intervals computed via bootstrap sampling."

### Slide 9: Feature Importance [DATA VISUALIZATION #2]
**Title:** What Drives the Predictions?
**Top Features (by importance):**
1. Dissolved Oxygen
2. pH
3. Nitrate
4. Temperature
5. Conductance
**SHAP Values:** Show per-prediction contributions
- Green bars = increases safety/WQI
- Orange bars = decreases safety/WQI
**Visual:** Horizontal bar chart of feature importances
**Speaker Notes:** "Dissolved oxygen is the most important predictor, followed by pH and nitrate. SHAP values let us explain individual predictions - showing exactly which features pushed the prediction up or down. This transparency is crucial for responsible AI."

### Slide 10: Live Demo
**Title:** Live Demo
**Content:**
- [Large text: "LIVE DEMONSTRATION"]
- ZIP codes to demo: 20001 (DC), 90210 (LA)
- Show: WQI calculation, ML prediction, SHAP explanation
**Backup:** Screenshots of app output if technical issues
**Speaker Notes:** "Now let me show you the application in action. I'll enter a ZIP code and walk you through the results..."

### Slide 11: Environmental Justice Analysis [DATA VISUALIZATION #3]
**Title:** Critical Limitation: 100% False Negative on Lead
**Content:**
- Tested on Flint & Jackson contamination scenarios
- **150 ppb lead (10× EPA limit) → WQI 100.0 "Excellent"**
- All 6 scenarios: 100% false negative rate
**Why:** NSF-style WQI does NOT include lead. ML models trained on WQI labels inherited this blind spot.
**Visual:** Table showing lead levels vs. predictions (all wrong)
**Speaker Notes:** "This is our most important finding. When we tested on water quality matching the Flint crisis, our system said 'Excellent' for water with 10 times the EPA lead limit. The model can't detect what it was never taught. This is a critical limitation we document explicitly."

### Slide 12: Positive & Negative Impact
**Title:** Impact Assessment
**Positive:**
- Makes government data accessible
- SHAP provides transparency
- Highlights EJ data gaps
- Educational tool
**Negative:**
- Cannot detect lead, bacteria, PFAS
- False sense of security
- Disproportionate risk to EJ communities
- Not a replacement for lab testing
**Visual:** Two-column layout with + and - headers
**Speaker Notes:** "Our system has real benefits for accessibility and education, but also real risks. The most dangerous impact is giving false confidence about water that's actually toxic. We're honest about both sides."

### Slide 13: Essential Question Answer
**Title:** How Can AI Amplify or Mitigate Bias?
**AMPLIFIES:**
- Labels inherit WQI's blind spot to lead
- European data misses US infrastructure decay
- AI confidence can mislead users
- Disproportionately affects Flint (~54% Black), Jackson (~80% Black)
**MITIGATES:**
- SHAP explanations show WHY
- Documented limitations
- Uncertainty quantification
- Acknowledging failure = responsible AI
**Visual:** Split layout showing both sides
**Speaker Notes:** "This is the essential question. Our project shows AI amplifies bias when models inherit blind spots from training labels. But we can mitigate bias through transparency, documented limitations, and honest acknowledgment of where our models fail. Discovering and documenting our 100% false negative rate on lead is itself a form of responsible AI."

### Slide 14: Next Steps
**Title:** What's Next
**Individual:**
- Sean: ML internships, Python certification, open source contributions
- Joseann: Data science courses, portfolio projects
- Lademi: Environmental policy internships, graduate programs
**Project:**
- Add lead data from Water Quality Portal
- Infrastructure risk indicators
- Partner with water utilities
- Community outreach
**Visual:** Roadmap or checklist format
**Speaker Notes:** "Each of us has specific, measurable next steps. For the project, the immediate priority is adding lead detection capability using data that's actually available in the Water Quality Portal - lead is a characteristic they track."

### Slide 15: Citations & Links
**Title:** References
**Citations (list all 8):**
1. Virginia Tech Flint Water Study (2015)
2. EPA Flint Response (2016-2019)
3. Hanna-Attisha et al. (2016) AJPH
4. CDC MMWR (2016)
5. ProPublica Jackson Investigation (2023)
6. EPA Jackson Response (2022)
7. WHO Lead Fact Sheet (2023)
8. Brown et al. (1970) NSF-WQI paper
**Data Sources:** Kaggle, USGS NWIS, Water Quality Portal
**GitHub:** https://github.com/seanesla/group11C
**Visual:** QR code to GitHub repo
**Speaker Notes:** "All our citations are listed here, along with our data sources and GitHub repository. Thank you for listening - we're happy to take questions."

## DESIGN SPECIFICATIONS

- **Color Scheme:** Blue (#1E88E5) and green (#43A047) for water theme, with orange (#FF7043) for warnings
- **Font:** Sans-serif (Roboto, Open Sans, or Arial)
- **Layout:** Minimal text, maximum 6 bullet points per slide
- **Visuals:** Include charts, diagrams, or images on every content slide
- **AI4ALL Logo:** Place on title slide
- **Consistent Headers:** Same position and style on all slides

Generate a complete Google Slides presentation with all content above, speaker notes for each slide, and professional visual design.
