// ==============================================================================
// DyTrStudio Feature Engineering & Multi-Column Selection Architecture
// Deep Support for Single-Feature, Multi-Feature, Feature Selection & Synthesis
// Zero Emojis Policy - High Academic Rigor - Bilingual RTL/LTR
// ==============================================================================

/**
 * Builds the composite formatted input string for a given dataset row
 * based on the active feature mode and engineering strategy.
 * 
 * @param {Object} row - The dataset row object
 * @param {Object} taskConfig - Task configuration object containing feature specifications
 * @returns {string} - The computed composite input string
 */
export function buildCompositeFeatureString(row, taskConfig = {}) {
  if (!row || typeof row !== 'object') return '';

  const featureMode = taskConfig.feature_mode || (taskConfig.feature_columns && taskConfig.feature_columns.length > 1 ? 'multiple' : 'single');
  const featureCols = Array.isArray(taskConfig.feature_columns) && taskConfig.feature_columns.length > 0
    ? taskConfig.feature_columns
    : (taskConfig.text_column ? [taskConfig.text_column] : [Object.keys(row)[0] || 'text']);

  const fe = taskConfig.feature_engineering || {};
  const strategy = fe.strategy || 'delimiter'; // 'delimiter' | 'custom_template' | 'derived'
  const delimiter = fe.delimiter !== undefined ? fe.delimiter : ' [SEP] ';
  const template = fe.template || '';
  const excludedCols = Array.isArray(fe.excluded_columns) ? fe.excluded_columns : [];

  // Filter out any explicitly excluded columns
  const activeCols = featureCols.filter((col) => !excludedCols.includes(col));

  if (activeCols.length === 0) {
    const firstKey = Object.keys(row).find((k) => !excludedCols.includes(k)) || Object.keys(row)[0];
    return String(row[firstKey] ?? '');
  }

  // Single feature mode
  if (featureMode === 'single' || activeCols.length === 1) {
    const colName = activeCols[0];
    let val = row[colName];
    if (val === null || val === undefined) val = '';
    return String(val);
  }

  // Custom Template Strategy (e.g. "{title} [SEP] {content}")
  if (strategy === 'custom_template' && template.trim().length > 0) {
    let result = template;
    for (const key of Object.keys(row)) {
      const val = row[key] !== null && row[key] !== undefined ? String(row[key]) : '';
      result = result.replaceAll(`{${key}}`, val);
    }
    return result;
  }

  // Derived / Calculated Strategy
  if (strategy === 'derived' && Array.isArray(fe.derived_features) && fe.derived_features.length > 0) {
    const parts = activeCols.map((col) => {
      const val = row[col] !== null && row[col] !== undefined ? String(row[col]) : '';
      return `${col}: ${val}`;
    });

    fe.derived_features.forEach((df) => {
      if (df && df.name && df.expression) {
        try {
          if (df.transformType === 'age_calc' && row[df.sourceCol]) {
            const birthYear = parseInt(row[df.sourceCol], 10);
            if (!isNaN(birthYear)) {
              const currentYear = new Date().getFullYear();
              parts.push(`${df.name}: ${currentYear - birthYear}`);
            }
          } else if (df.transformType === 'tag_prefix') {
            parts.push(`[${df.name.toUpperCase()}] ${row[df.sourceCol] || ''}`);
          }
        } catch {
          // Graceful fallback
        }
      }
    });

    return parts.join(delimiter);
  }

  // Standard Delimiter Strategy (e.g. "title [SEP] content [SEP] author")
  const parts = activeCols.map((col) => {
    const val = row[col];
    return val !== null && val !== undefined ? String(val).trim() : '';
  }).filter((v) => v.length > 0);

  return parts.join(delimiter);
}

/**
 * Generates PyTorch & Pandas feature engineering code for standalone scripts
 */
export function generateFeaturePipelinePythonCode(taskConfig = {}) {
  const featureMode = taskConfig.feature_mode || 'single';
  const featureCols = taskConfig.feature_columns || [taskConfig.text_column || 'text'];
  const fe = taskConfig.feature_engineering || {};
  const delimiter = fe.delimiter || ' [SEP] ';
  const excluded = fe.excluded_columns || [];

  const activeCols = featureCols.filter((c) => !excluded.includes(c));

  if (featureMode === 'single' || activeCols.length <= 1) {
    const col = activeCols[0] || 'text';
    return `# 1. Single-Feature Extraction Pipeline
def extract_input_texts(df):
    text_col = "${col}"
    return df[text_col].astype(str).fillna("").values
`;
  }

  const colsRepr = JSON.stringify(activeCols);
  return `# 1. Multi-Feature & Feature Engineering Pipeline
def extract_input_texts(df):
    feature_cols = ${colsRepr}
    delimiter = "${delimiter}"
    # Feature Concatenation with Structured Delimiter
    return df[feature_cols].astype(str).fillna("").agg(delimiter.join, axis=1).values
`;
}
