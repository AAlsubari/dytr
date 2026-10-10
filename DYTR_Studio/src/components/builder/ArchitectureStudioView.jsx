// ==============================================================================
// DyTrStudio Model Selection & Architecture Studio (مختبر بناء واختيار النموذج)
// Stage 03 of AI Development Lifecycle (Problem -> Academy -> DataPrep -> ModelSelection -> Arena)
// Foundational Pedagogical Workflow:
// 1. Input/Output Contract: What goes in (Input) -> What comes out (Expected Output)
// 2. Model Taxonomy: Classification, Generation, Entity Tagging, Numeric Scoring
// 3. Model Strategy: From Scratch vs Pretrained Fine-Tuning
// 4. Model Sizing: Proportional to dataset volume (Anti-Overfitting)
// Zero Emojis Policy - High Academic Rigor - 3-Layer TermTooltip Integration
// ==============================================================================

import React, { useState, useEffect, useMemo } from 'react';
import { useStudioStore } from '../../store/studioStore';
import { useDatasetStore } from '../../store/useDatasetStore';
import { academicDictionary } from '../../utils/academicDictionary';
import { api } from '../../utils/api';
import TermTooltip from '../common/TermTooltip';
import SectionCard from '../common/SectionCard';
import FeatureEngineeringSuite from '../common/FeatureEngineeringSuite';
import { buildCompositeFeatureString } from '../../utils/featureEngineering';
import { ArrowRight, ArrowLeft, Check } from 'lucide-react';

export default function ArchitectureStudioView() {
  const {
    language,
    modelConfig,
    updateModelConfig,
    modelEstimate,
    connectionMode,
    setActiveModule,
    loadTrainedModelCheckpoint,
  } = useStudioStore();

  const { currentDataset, datasetName } = useDatasetStore();

  const d = academicDictionary[language] || academicDictionary.en;
  const isAr = language === 'ar';

  // 1. Core Model States
  const [selectedTaskType, setSelectedTaskType] = useState(
    modelConfig.tasks?.[0]?.task_type || 'sentence_classification'
  );

  const [modelStrategy, setModelStrategy] = useState(
    modelConfig.pretrained_backbone && modelConfig.pretrained_backbone !== 'custom_dytr'
      ? 'pretrained'
      : 'scratch'
  );

  const [selectedPretrainedModel, setSelectedPretrainedModel] = useState(
    modelConfig.pretrained_backbone && modelConfig.pretrained_backbone !== 'custom_dytr'
      ? modelConfig.pretrained_backbone
      : 'bert'
  );

  const [selectedSizeArchetype, setSelectedSizeArchetype] = useState('balanced');
  const [isCustomSizing, setIsCustomSizing] = useState(false);
  const [loraRank, setLoraRank] = useState(8);
  const [loraAlpha, setLoraAlpha] = useState(16.0);
  const [customDropout, setCustomDropout] = useState(0.1);

  // Custom Local Model Upload & Checkpoint State
  const [customModelFile, setCustomModelFile] = useState(null);
  const [customModelConfigFile, setCustomModelConfigFile] = useState(null);
  const [customModelConfigName, setCustomModelConfigName] = useState('');
  const [customModelConfigRaw, setCustomModelConfigRaw] = useState('');
  const [customModelConfigParsed, setCustomModelConfigParsed] = useState(null);
  const [autoDetectedFromConfig, setAutoDetectedFromConfig] = useState(false);
  const [customArchitecturePattern, setCustomArchitecturePattern] = useState('dytr_native'); // 'dytr_native' | 'huggingface_standard' | 'pytorch_state_dict' | 'onnx_runtime'
  const [showConfigJsonViewer, setShowConfigJsonViewer] = useState(false);
  const [customModelPath, setCustomModelPath] = useState('checkpoint_local.safetensors');
  const [customModelFormat, setCustomModelFormat] = useState('safetensors'); // 'safetensors' | 'pytorch' | 'onnx' | 'huggingface'
  const [customModelFamily, setCustomModelFamily] = useState('encoder'); // 'encoder' | 'decoder' | 'seq2seq' | 'custom'
  const [customModelParams, setCustomModelParams] = useState({
    embed_dim: 768,
    num_layers: 12,
    num_heads: 12,
    vocab_size: 30522,
    max_seq_len: 512,
  });
  const [isDragOverModel, setIsDragOverModel] = useState(false);
  const [isDragOverConfig, setIsDragOverConfig] = useState(false);
  const [isUploadingModel, setIsUploadingModel] = useState(false);
  const [uploadSuccessMessage, setUploadSuccessMessage] = useState('');

  // Input & Output Column Binding & Feature Engineering State
  const [selectedTextCol, setSelectedTextCol] = useState(
    modelConfig.tasks?.[0]?.text_column || ''
  );
  const [selectedLabelCol, setSelectedLabelCol] = useState(
    modelConfig.tasks?.[0]?.label_column || ''
  );
  const [featureMode, setFeatureMode] = useState(
    modelConfig.tasks?.[0]?.feature_mode || 'single'
  );
  const [featureColumns, setFeatureColumns] = useState(
    modelConfig.tasks?.[0]?.feature_columns || (modelConfig.tasks?.[0]?.text_column ? [modelConfig.tasks[0].text_column] : [])
  );
  const [featureEngineering, setFeatureEngineering] = useState(
    modelConfig.tasks?.[0]?.feature_engineering || {
      strategy: 'delimiter',
      delimiter: ' [SEP] ',
      template: '',
      excluded_columns: [],
      derived_features: [],
    }
  );

  // Live Generated Python Code
  const [livePythonCode, setLivePythonCode] = useState('');
  const [copySuccess, setCopySuccess] = useState(false);

  // Helper: Smart Parser for Companion Config JSON (dytr_model_config.json / config.json)
  const parseAndApplyConfigJson = (jsonString, fileName) => {
    try {
      const parsed = typeof jsonString === 'string' ? JSON.parse(jsonString) : jsonString;
      setCustomModelConfigParsed(parsed);
      setCustomModelConfigRaw(typeof jsonString === 'string' ? jsonString : JSON.stringify(jsonString, null, 2));
      setCustomModelConfigName(fileName || 'model_config.json');

      const d_model =
        parsed.hidden_size ||
        parsed.embed_dim ||
        parsed.d_model ||
        parsed.n_embd ||
        parsed.embedding_dim ||
        customModelParams.embed_dim;
      const layers =
        parsed.num_hidden_layers ||
        parsed.num_layers ||
        parsed.n_layer ||
        parsed.layers ||
        parsed.num_encoder_layers ||
        customModelParams.num_layers;
      const heads =
        parsed.num_attention_heads ||
        parsed.num_heads ||
        parsed.n_head ||
        parsed.heads ||
        parsed.num_encoder_attention_heads ||
        customModelParams.num_heads;
      const vocab =
        parsed.vocab_size ||
        parsed.vocabulary_size ||
        customModelParams.vocab_size;
      const max_seq =
        parsed.max_position_embeddings ||
        parsed.n_positions ||
        parsed.max_seq_len ||
        parsed.max_sequence_length ||
        parsed.seq_len ||
        customModelParams.max_seq_len;

      let detectedFamily = customModelFamily;
      const archStr = Array.isArray(parsed.architectures)
        ? parsed.architectures.join(' ')
        : (parsed.model_type || parsed.model_family || '');
      const archLower = archStr.toLowerCase();
      if (
        archLower.includes('causal') ||
        archLower.includes('gpt') ||
        archLower.includes('llama') ||
        archLower.includes('qwen') ||
        archLower.includes('mistral')
      ) {
        detectedFamily = 'decoder';
      } else if (
        archLower.includes('t5') ||
        archLower.includes('bart') ||
        archLower.includes('conditional') ||
        archLower.includes('seq2seq')
      ) {
        detectedFamily = 'seq2seq';
      } else if (
        archLower.includes('bert') ||
        archLower.includes('encoder') ||
        archLower.includes('classification') ||
        archLower.includes('maskedlm')
      ) {
        detectedFamily = 'encoder';
      }

      let detectedPattern = customArchitecturePattern;
      if (parsed.dytr_version || parsed.framework === 'dytr' || fileName?.includes('dytr')) {
        detectedPattern = 'dytr_native';
      } else if (parsed.transformers_version || parsed.architectures || parsed.model_type) {
        detectedPattern = 'huggingface_standard';
      } else if (parsed.torch_version || fileName?.endsWith('.pt')) {
        detectedPattern = 'pytorch_state_dict';
      }

      const updatedParams = {
        embed_dim: Number(d_model) || 768,
        num_layers: Number(layers) || 12,
        num_heads: Number(heads) || 12,
        vocab_size: Number(vocab) || 30522,
        max_seq_len: Number(max_seq) || 512,
      };

      setCustomModelParams(updatedParams);
      setCustomModelFamily(detectedFamily);
      setCustomArchitecturePattern(detectedPattern);
      setAutoDetectedFromConfig(true);

      updateModelConfig({
        embed_dim: updatedParams.embed_dim,
        num_layers: updatedParams.num_layers,
        num_heads: updatedParams.num_heads,
        vocab_size: updatedParams.vocab_size,
        max_seq_len: updatedParams.max_seq_len,
      });

      return updatedParams;
    } catch (err) {
      console.warn('Failed to parse companion config JSON:', err);
      return null;
    }
  };

  const processConfigFile = (file) => {
    setCustomModelConfigFile(file);
    const reader = new FileReader();
    reader.onload = (event) => {
      const text = event.target.result;
      const parsed = parseAndApplyConfigJson(text, file.name);
      if (parsed) {
        setUploadSuccessMessage(
          isAr
            ? `تم تحليل وضبط المعمارية بنجاح من ملف التهيئة: ${file.name}`
            : `Successfully parsed & applied architecture from: ${file.name}`
        );
        setTimeout(() => setUploadSuccessMessage(''), 4000);
      }
    };
    reader.readAsText(file);
  };

  const processUploadedFiles = (files) => {
    setIsUploadingModel(true);
    let weightFile = null;
    let configFile = null;

    for (const f of files) {
      const name = f.name.toLowerCase();
      if (name.endsWith('.json')) {
        configFile = f;
      } else if (
        name.endsWith('.safetensors') ||
        name.endsWith('.pt') ||
        name.endsWith('.pth') ||
        name.endsWith('.bin') ||
        name.endsWith('.onnx') ||
        name.endsWith('.zip')
      ) {
        weightFile = f;
      }
    }

    // If only JSON is uploaded in the main drop zone
    if (!weightFile && configFile) {
      processConfigFile(configFile);
      setIsUploadingModel(false);
      return;
    }

    // If weight file is present
    if (weightFile) {
      const fileName = weightFile.name.toLowerCase();
      let detectedFormat = 'safetensors';
      if (fileName.endsWith('.pt') || fileName.endsWith('.pth') || fileName.endsWith('.bin')) detectedFormat = 'pytorch';
      else if (fileName.endsWith('.onnx')) detectedFormat = 'onnx';
      else if (fileName.endsWith('.json') || fileName.endsWith('.zip')) detectedFormat = 'huggingface';

      let detectedFamily = 'encoder';
      if (fileName.includes('gpt') || fileName.includes('llama') || fileName.includes('qwen') || fileName.includes('mistral') || fileName.includes('causal')) {
        detectedFamily = 'decoder';
      } else if (fileName.includes('t5') || fileName.includes('bart') || fileName.includes('seq2seq')) {
        detectedFamily = 'seq2seq';
      }

      setCustomModelFile({
        name: weightFile.name,
        size: weightFile.size,
        type: weightFile.type || detectedFormat,
        lastModified: weightFile.lastModified,
      });
      setCustomModelPath(weightFile.name);
      setCustomModelFormat(detectedFormat);
      setCustomModelFamily(detectedFamily);

      // Also parse companion config if dropped simultaneously
      if (configFile) {
        processConfigFile(configFile);
      }

      setTimeout(() => {
        setIsUploadingModel(false);
        setUploadSuccessMessage(
          isAr
            ? `تم فحص وتثبيت النموذج المحلي بنجاح: ${weightFile.name}${configFile ? ` مع ملف التهيئة (${configFile.name})` : ''}`
            : `Successfully loaded local checkpoint: ${weightFile.name}${configFile ? ` with companion config (${configFile.name})` : ''}`
        );
        setTimeout(() => setUploadSuccessMessage(''), 4000);

        loadTrainedModelCheckpoint({
          backbone: 'custom_local',
          custom_model_path: weightFile.name,
          name: weightFile.name,
          format: detectedFormat,
          embed_dim: customModelParams.embed_dim,
          num_layers: customModelParams.num_layers,
          num_heads: customModelParams.num_heads,
          vocab_size: customModelParams.vocab_size,
          max_seq_len: customModelParams.max_seq_len,
        });
      }, 400);
    } else {
      setIsUploadingModel(false);
    }
  };

  // Drag & Drop / File Input Processing for Custom Local Model & Companion Config
  const handleModelFileDrop = (e) => {
    e.preventDefault();
    setIsDragOverModel(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      processUploadedFiles(Array.from(e.dataTransfer.files));
    }
  };

  const handleConfigFileDrop = (e) => {
    e.preventDefault();
    setIsDragOverConfig(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const file = e.dataTransfer.files[0];
      processConfigFile(file);
    }
  };

  const handleModelFileChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      processUploadedFiles(Array.from(e.target.files));
    }
  };

  const handleConfigFileChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      processConfigFile(e.target.files[0]);
    }
  };

  // 2. Dataset Normalization and Smart Column Auto-Detection
  const datasetCols = useMemo(() => {
    if (!currentDataset || !Array.isArray(currentDataset) || currentDataset.length === 0) return [];
    const firstRow = currentDataset[0];
    if (!firstRow || typeof firstRow !== 'object' || Array.isArray(firstRow)) return [];
    return Object.keys(firstRow).filter((key) => isNaN(Number(key)));
  }, [currentDataset]);

  const datasetAnalysis = useMemo(() => {
    if (!currentDataset || !Array.isArray(currentDataset) || currentDataset.length === 0) {
      return {
        rowsCount: 0,
        colsCount: 0,
        textCols: [],
        labelCols: [],
        uniqueClasses: [],
        defaultTextCol: '',
        defaultLabelCol: '',
        recommendedSize: 'balanced',
      };
    }

    const rowsCount = currentDataset.length;
    const colsCount = datasetCols.length;

    // Score and categorize candidate columns
    const colStats = datasetCols.map((col) => {
      const sample = currentDataset.slice(0, 50).map((r) => String(r[col] !== undefined && r[col] !== null ? r[col] : ''));
      const avgLen = sample.reduce((acc, s) => acc + s.length, 0) / Math.max(1, sample.length);
      const hasSpaces = sample.filter((s) => s.includes(' ')).length / Math.max(1, sample.length);
      const uniqueVals = new Set(sample.filter(Boolean));
      const colLower = col.toLowerCase();

      // Text scoring (longer strings with spaces)
      let textScore = avgLen * 1.5 + hasSpaces * 30;
      if (/text|sms|message|body|content|review|sentence|headline|comment|v2|desc|input|prompt/i.test(colLower)) textScore += 60;
      if (/label|class|category|target|sentiment|id|count|num/i.test(colLower)) textScore -= 40;

      // Label scoring (discrete values)
      let labelScore = 0;
      if (uniqueVals.size >= 1 && uniqueVals.size <= 50) labelScore += (50 - uniqueVals.size);
      if (/label|class|category|target|sentiment|tag|topic|v1|type|output/i.test(colLower)) labelScore += 60;
      if (/id|count|index/i.test(colLower)) labelScore -= 50;
      if (avgLen > 60) labelScore -= 40;

      return {
        col,
        avgLen,
        uniqueCount: uniqueVals.size,
        textScore,
        labelScore,
      };
    });

    const sortedByText = [...colStats].sort((a, b) => b.textScore - a.textScore);
    const sortedByLabel = [...colStats].sort((a, b) => b.labelScore - a.labelScore);

    const bestTextCol = sortedByText[0]?.col || datasetCols[0] || '';
    let bestLabelCol = sortedByLabel.find((c) => c.col !== bestTextCol)?.col || sortedByLabel[0]?.col || datasetCols[datasetCols.length - 1] || '';

    // If dataset has 2+ columns, ensure text & label are strictly distinct
    if (datasetCols.length >= 2 && bestTextCol === bestLabelCol) {
      bestLabelCol = datasetCols.find((c) => c !== bestTextCol) || datasetCols[1];
    }

    const textCols = sortedByText.map((c) => c.col);
    const labelCols = sortedByLabel.map((c) => c.col);

    // Calculate unique classes for active label column
    const activeLabel = (selectedLabelCol && datasetCols.includes(selectedLabelCol)) ? selectedLabelCol : bestLabelCol;
    let uniqueClasses = [];
    if (activeLabel) {
      uniqueClasses = Array.from(
        new Set(currentDataset.map((r) => String(r[activeLabel] !== undefined && r[activeLabel] !== null ? r[activeLabel] : '').trim()).filter(Boolean))
      );
    }

    let recommendedSize = 'balanced';
    if (rowsCount < 1000) recommendedSize = 'edge';
    else if (rowsCount > 20000) recommendedSize = 'deep';

    return {
      rowsCount,
      colsCount,
      textCols,
      labelCols,
      uniqueClasses,
      defaultTextCol: bestTextCol,
      defaultLabelCol: bestLabelCol,
      recommendedSize,
    };
  }, [currentDataset, datasetCols, selectedLabelCol]);

  // Synchronize default column mappings on initial load and sync to store
  const syncModelTaskHead = (
    textCol,
    labelCol,
    taskType,
    classesList,
    featMode,
    featCols,
    featEng
  ) => {
    const finalClasses = classesList && classesList.length > 0
      ? classesList
      : datasetAnalysis.uniqueClasses && datasetAnalysis.uniqueClasses.length > 0
      ? datasetAnalysis.uniqueClasses
      : ['ham', 'spam'];

    const currentType = taskType || selectedTaskType || 'sentence_classification';
    const numClasses = currentType === 'causal_lm'
      ? (modelConfig.vocab_size || 30522)
      : currentType === 'regression_scoring'
      ? 1
      : finalClasses.length;

    const singleTask = {
      task_name: 'task_classification',
      task_type: currentType === 'regression_scoring' ? 'sentence_classification' : currentType,
      strategy:
        currentType === 'causal_lm'
          ? 'CAUSAL_LM'
          : currentType === 'token_classification'
          ? 'TOKEN_CLASSIFICATION'
          : 'SENTENCE_CLASSIFICATION',
      num_classes: numClasses,
      classes: finalClasses,
      text_column: textCol || selectedTextCol || datasetAnalysis.defaultTextCol || 'text',
      label_column: labelCol || selectedLabelCol || datasetAnalysis.defaultLabelCol || 'label',
      max_len: modelConfig.max_seq_len || 128,
      feature_mode: featMode || featureMode || 'single',
      feature_columns: featCols || featureColumns || [textCol || selectedTextCol || 'text'],
      feature_engineering: featEng || featureEngineering || {
        strategy: 'delimiter',
        delimiter: ' [SEP] ',
        template: '',
        excluded_columns: [],
        derived_features: [],
      },
    };

    // Preserve any existing multi-task / continual heads if present
    const otherTasks = (modelConfig.tasks || []).filter((_, idx) => idx > 0);
    updateModelConfig({ tasks: [singleTask, ...otherTasks] });
  };

  const handleFeatureConfigChange = (newFeatConfig) => {
    const nextTextCol = newFeatConfig.text_column || selectedTextCol;
    const nextLabelCol = newFeatConfig.label_column || selectedLabelCol;
    const nextMode = newFeatConfig.feature_mode || featureMode;
    const nextCols = newFeatConfig.feature_columns || featureColumns;
    const nextFE = newFeatConfig.feature_engineering || featureEngineering;

    if (newFeatConfig.text_column) setSelectedTextCol(newFeatConfig.text_column);
    if (newFeatConfig.label_column) setSelectedLabelCol(newFeatConfig.label_column);
    if (newFeatConfig.feature_mode) setFeatureMode(newFeatConfig.feature_mode);
    if (newFeatConfig.feature_columns) setFeatureColumns(newFeatConfig.feature_columns);
    if (newFeatConfig.feature_engineering) setFeatureEngineering(newFeatConfig.feature_engineering);

    let activeClasses = datasetAnalysis.uniqueClasses;
    if (newFeatConfig.label_column && currentDataset) {
      activeClasses = Array.from(new Set(currentDataset.map((r) => String(r[newFeatConfig.label_column] || '').trim()).filter(Boolean)));
    }

    syncModelTaskHead(
      nextTextCol,
      nextLabelCol,
      selectedTaskType,
      activeClasses,
      nextMode,
      nextCols,
      nextFE
    );
  };

  useEffect(() => {
    if (datasetCols.length === 0) return;

    let textToSet = selectedTextCol;
    let labelToSet = selectedLabelCol;

    const isTextValid = textToSet && datasetCols.includes(textToSet);
    const isLabelValid = labelToSet && datasetCols.includes(labelToSet);

    if (!isTextValid) {
      textToSet = datasetAnalysis.defaultTextCol;
      setSelectedTextCol(textToSet);
    }
    if (!isLabelValid) {
      labelToSet = datasetAnalysis.defaultLabelCol;
      setSelectedLabelCol(labelToSet);
    }

    // Ensure distinct if dataset has multiple columns
    if (datasetCols.length >= 2 && textToSet === labelToSet) {
      if (datasetAnalysis.defaultTextCol !== datasetAnalysis.defaultLabelCol) {
        textToSet = datasetAnalysis.defaultTextCol;
        labelToSet = datasetAnalysis.defaultLabelCol;
      } else {
        labelToSet = datasetCols.find((c) => c !== textToSet) || labelToSet;
      }
      setSelectedTextCol(textToSet);
      setSelectedLabelCol(labelToSet);
    }

    // Protection rule: Only auto-initialize tasks if modelConfig.tasks is completely empty
    // Never overwrite an already configured or trained model unintentionally upon entering the page
    if (!modelConfig.tasks || modelConfig.tasks.length === 0) {
      if (textToSet && labelToSet) {
        const activeClasses = datasetAnalysis.uniqueClasses?.length > 0
          ? datasetAnalysis.uniqueClasses
          : Array.from(new Set(currentDataset.map((r) => String(r[labelToSet] || '').trim()).filter(Boolean)));
        syncModelTaskHead(textToSet, labelToSet, selectedTaskType, activeClasses);
      }
    }
  }, [datasetAnalysis, datasetCols, modelConfig.tasks]);

  const handleTextColChange = (col) => {
    setSelectedTextCol(col);
    syncModelTaskHead(col, selectedLabelCol, selectedTaskType, datasetAnalysis.uniqueClasses);
  };

  const handleLabelColChange = (col) => {
    setSelectedLabelCol(col);
    // recalculate classes for the new label column
    let newClasses = [];
    if (currentDataset && col) {
      newClasses = Array.from(new Set(currentDataset.map((r) => String(r[col] || '').trim()).filter(Boolean)));
    }
    syncModelTaskHead(selectedTextCol, col, selectedTaskType, newClasses);
  };

  // Target class count
  const targetNumClasses = useMemo(() => {
    if (!datasetAnalysis.uniqueClasses || datasetAnalysis.uniqueClasses.length === 0) return 2;
    return Math.max(2, datasetAnalysis.uniqueClasses.length);
  }, [datasetAnalysis.uniqueClasses]);

  // Sample row for interactive transformation mapping
  const sampleRow = useMemo(() => {
    if (!currentDataset || currentDataset.length === 0) return null;
    const row = currentDataset[0] || {};
    const actualTextCol = (selectedTextCol && datasetCols.includes(selectedTextCol))
      ? selectedTextCol
      : (datasetAnalysis.defaultTextCol || Object.keys(row)[1] || Object.keys(row)[0] || 'text');
    const actualLabelCol = (selectedLabelCol && datasetCols.includes(selectedLabelCol))
      ? selectedLabelCol
      : (datasetAnalysis.defaultLabelCol || Object.keys(row)[0] || 'label');

    return {
      text: actualTextCol ? String(row[actualTextCol] !== undefined && row[actualTextCol] !== null ? row[actualTextCol] : '') : '',
      label: actualLabelCol ? String(row[actualLabelCol] !== undefined && row[actualLabelCol] !== null ? row[actualLabelCol] : '') : '',
      actualTextCol,
      actualLabelCol,
      rawRow: row,
    };
  }, [currentDataset, selectedTextCol, selectedLabelCol, datasetCols, datasetAnalysis]);

  // 3. Model Taxonomy & Archetypes Definition (Clean & Minimalist with 3-Layer Tooltips)
  const modelTaxonomy = [
    {
      id: 'sentence_classification',
      tooltipKey: 'task_classification',
      name_ar: 'تصنيف النصوص (Classification)',
      name_en: 'Text Classification',
      inputLabel_ar: 'نص',
      inputLabel_en: 'Text',
      outputLabel_ar: 'فئة',
      outputLabel_en: 'Category',
      desc_ar: 'تحديد فئة النص ككل (مثل: تحليل المشاعر إيجابي/سلبي، أو فرز وتصنيف الأخبار).',
      desc_en: 'Categorize entire text sequences (e.g. sentiment polarity, topic classification).',
      outputTag_ar: `فئة (${targetNumClasses} فئات)`,
      outputTag_en: `${targetNumClasses} Classes`,
    },
    {
      id: 'causal_lm',
      tooltipKey: 'task_generation',
      name_ar: 'توليد النصوص (Text Generation)',
      name_en: 'Text Generation',
      inputLabel_ar: 'نص',
      inputLabel_en: 'Text',
      outputLabel_ar: 'نص مكمل',
      outputLabel_en: 'Next Text',
      desc_ar: 'إكمال الجمل وتوليد نصوص جديدة متتابعة (مثل روبوتات المحادثة وكتابة الردود).',
      desc_en: 'Complete sentences and generate ongoing coherent text (e.g. conversational chatbots).',
      outputTag_ar: 'نص مولد',
      outputTag_en: 'Generated Text',
    },
    {
      id: 'token_classification',
      tooltipKey: 'task_ner',
      name_ar: 'استخراج الكيانات (Named Entities)',
      name_en: 'Named Entity Recognition',
      inputLabel_ar: 'نص',
      inputLabel_en: 'Text',
      outputLabel_ar: 'وسوم كلمات',
      outputLabel_en: 'Word Tags',
      desc_ar: 'استخراج وتصنيف الأسماء، المدن، التواريخ، والشركات من داخل النصوص.',
      desc_en: 'Extract and label person names, locations, dates, and organizations in text.',
      outputTag_ar: 'وسوم الكلمات',
      outputTag_en: 'Entity Tags',
    },
    {
      id: 'regression_scoring',
      tooltipKey: 'task_regression',
      name_ar: 'التنبؤ الرقمي والتقييم (Scoring)',
      name_en: 'Numeric Regression & Scoring',
      inputLabel_ar: 'نص',
      inputLabel_en: 'Text',
      outputLabel_ar: 'رقم / تقييم',
      outputLabel_en: 'Numeric Score',
      desc_ar: 'تقدير رقم أو درجة تقييمية مستمرة بناءً على النص (مثل تقييم 1 إلى 5 نجوم).',
      desc_en: 'Estimate a continuous number or rating score from text (e.g. 1 to 5 stars).',
      outputTag_ar: 'رقم / تقييم',
      outputTag_en: 'Numeric Score',
    },
  ];

  // 4. Sizing Archetypes Definition
  const sizeArchetypes = {
    edge: {
      id: 'edge',
      name_ar: 'النموذج الخفيف (Lightweight)',
      name_en: 'Lightweight Model',
      badge_ar: 'سريع جدا واقتصادي',
      badge_en: 'Ultra-Fast & Low Memory',
      desc_ar: 'معمارية مضغوطة من طبقتين، مثالية للبيانات الصغيرة (أقل من 1,000 صف) لتفادي فرط التخصيص والعمل على الأجهزة المحدودة.',
      desc_en: 'Compact 2-layer network ideal for small datasets (<1k rows) and low-power hardware to avoid overfitting.',
      layers: 2,
      d_model: 128,
      heads: 2,
      max_seq: 128,
      suitableFor_ar: 'بيانات صغيرة (أقل من 1,000 صف) / أجهزة عادية',
      suitableFor_en: 'Small data (<1k rows) / Standard devices',
    },
    balanced: {
      id: 'balanced',
      name_ar: 'النموذج القياسي المتوازن (Balanced)',
      name_en: 'Standard Balanced Model',
      badge_ar: 'موصى به لمعظم المهام',
      badge_en: 'Recommended Baseline',
      desc_ar: 'معمارية متوازنة من 4 طبقات، تضمن دقة تمثيل عالية وسرعة تدريب ممتازة لمجموعات البيانات المتوسطة.',
      desc_en: 'Optimal 4-layer balance of representational capacity, training speed, and generalization.',
      layers: 4,
      d_model: 256,
      heads: 4,
      max_seq: 256,
      suitableFor_ar: 'بيانات متوسطة (1,000 إلى 20,000 صف)',
      suitableFor_en: 'Medium datasets (1k to 20k rows)',
    },
    deep: {
      id: 'deep',
      name_ar: 'النموذج عالي السعة (High-Capacity)',
      name_en: 'Deep High-Capacity Model',
      badge_ar: 'سعة استيعابية قصوى',
      badge_en: 'Deep Representation',
      desc_ar: 'معمارية موسعة من 8 طبقات مصممة للمهام المعقدة وتصنيف النصوص الطويلة ومجموعات البيانات الكبيرة.',
      desc_en: 'Expanded 8-layer network engineered for complex contextual relations and massive datasets (>20k rows).',
      layers: 8,
      d_model: 512,
      heads: 8,
      max_seq: 512,
      suitableFor_ar: 'بيانات ضخمة (أكثر من 20,000 صف)',
      suitableFor_en: 'Massive corpora (>20k rows)',
    },
  };

  // Pretrained Foundation Models Catalog (Including Custom Local Model Upload)
  const pretrainedCatalog = useMemo(() => ({
    bert: {
      id: 'bert',
      name: 'BERT / RoBERTa (Encoder)',
      tooltipKey: 'backbone_bert',
      category_ar: 'نماذج الفهم والتصنيف',
      category_en: 'Encoder Backbone',
      bestFor_ar: 'تصنيف المشاعر، تصنيف الموضوعات، والبحث الدلالي',
      bestFor_en: 'Sentiment Analysis, Topic Classification, Semantic Search',
      embed_dim: 768,
      num_layers: 12,
      num_heads: 12,
      desc_ar: 'نموذج محول ثنائي الاتجاه لفهم الكلمات في سياقها الكامل وتصنيف النصوص بدقة.',
      desc_en: 'Bidirectional encoder model for sentence comprehension and text classification.',
    },
    arabert: {
      id: 'arabert',
      name: 'AraBERT / CAMeL (Arabic Encoder)',
      tooltipKey: 'backbone_arabert',
      category_ar: 'نماذج مخصصة للغة العربية',
      category_en: 'Arabic Specialized Encoder',
      bestFor_ar: 'معالجة النصوص العربية، اللهجات المحلية، وتحليل المشاعر العربي',
      bestFor_en: 'Arabic NLP, Dialectal Processing, Arabic Sentiment',
      embed_dim: 768,
      num_layers: 12,
      num_heads: 12,
      desc_ar: 'مدرب مسبقاً على مليارات النصوص العربية لفهم الصرف والتراكيب واللهجات.',
      desc_en: 'Pretrained on massive Arabic corpora for morphology, syntax, and dialectal nuances.',
    },
    gpt2: {
      id: 'gpt2',
      name: 'GPT-2 / LLaMA (Decoder)',
      tooltipKey: 'backbone_gpt',
      category_ar: 'نماذج التوليد اللغوي',
      category_en: 'Decoder Backbone',
      bestFor_ar: 'توليد النصوص، إكمال الجمل، والمحادثة الذكية',
      bestFor_en: 'Causal Text Generation, Completion, Dialogue',
      embed_dim: 768,
      num_layers: 12,
      num_heads: 12,
      desc_ar: 'نموذج لغوي سببي مصمم لتوليد النصوص كلمة بكلمة بناءً على السياق السابق.',
      desc_en: 'Autoregressive generative backbone with causal attention for continuous text generation.',
    },
    t5: {
      id: 't5',
      name: 'T5 / BART (Seq2Seq)',
      tooltipKey: 'backbone_t5',
      category_ar: 'نماذج التحويل والترجمة',
      category_en: 'Seq2Seq Backbone',
      bestFor_ar: 'الترجمة الآلية، التلخيص، وإعادة الصياغة',
      bestFor_en: 'Machine Translation, Summarization, Paraphrasing',
      embed_dim: 512,
      num_layers: 6,
      num_heads: 8,
      desc_ar: 'معمارية مشفر وفاك تشفير متكاملة تستلم نصاً وتنتج نصاً جديداً للترجمة والتلخيص.',
      desc_en: 'Unified encoder-decoder architecture ideal for translation, summarization, and rewriting.',
    },
    custom_local: {
      id: 'custom_local',
      name: isAr ? 'رفع نموذج محلي (Local Model Upload)' : 'Upload Local Model / Checkpoint',
      tooltipKey: 'custom_local_model',
      category_ar: 'نماذج وأوزان مخصصة ومحلية',
      category_en: 'Custom Local & Uploaded Weights',
      bestFor_ar: 'نقاط حفظ خاصة، أوزان .safetensors/.pt، ونماذج مدربة محلياً',
      bestFor_en: 'In-house trained models, local .safetensors / .pt checkpoints',
      embed_dim: customModelParams.embed_dim || 768,
      num_layers: customModelParams.num_layers || 12,
      num_heads: customModelParams.num_heads || 12,
      desc_ar: 'رفع واستيراد ملفات الأوزان ونقاط الحفظ المحلية لاستخدامها كعمود فقري وتكييفها بالـ LoRA.',
      desc_en: 'Import and upload custom local weights (.safetensors, .pt, .onnx) as your foundational backbone.',
    },
  }), [isAr, customModelParams]);

  // Scientific LoRA Benchmark Presets (Crisp & Simplified)
  const loraPresets = [
    {
      id: 'academic',
      name_ar: 'المعيار القياسي (Standard Baseline)',
      name_en: 'Standard Baseline (Hu et al.)',
      rank: 8,
      alpha: 16.0,
      desc_ar: 'الخيار الموصى به لمهام تصنيف وفهم النصوص (تحديث < 0.3% من الأوزان).',
      desc_en: 'Recommended standard for classification and language understanding (< 0.3% trainable).',
      trainableStr: '< 0.3%',
    },
    {
      id: 'lightweight',
      name_ar: 'فائق الخفة والسرعة (Ultra-Light)',
      name_en: 'Ultra-Lightweight',
      rank: 4,
      alpha: 8.0,
      desc_ar: 'أقل استهلاك لذاكرة كرت الشاشة وأسرع تدريب على الأجهزة المحدودة.',
      desc_en: 'Minimal VRAM consumption and fast training on low-resource hardware.',
      trainableStr: '< 0.15%',
    },
    {
      id: 'generative',
      name_ar: 'عالي السعة (High-Capacity)',
      name_en: 'High-Capacity Generative',
      rank: 16,
      alpha: 32.0,
      desc_ar: 'سعة تمثيل أكبر للمهام التوليدية، التلخيص، والترجمة المعقدة.',
      desc_en: 'Expanded capacity for text generation, summarization, and translation.',
      trainableStr: '< 0.6%',
    },
    {
      id: 'deep_domain',
      name_ar: 'تكييف تخصصي عميق (Deep Domain)',
      name_en: 'Deep Domain Adaptation',
      rank: 32,
      alpha: 64.0,
      desc_ar: 'للمجالات المتخصصة بدقة (المصطلحات الطبية، القانونية، واللهجات الدقيقة).',
      desc_en: 'Designed for steep domain shifts (medical, legal, extreme dialectal texts).',
      trainableStr: '< 1.1%',
    },
  ];

  // Handlers for Decisions
  const handleTaskTypeChange = (taskType) => {
    setSelectedTaskType(taskType);
    const updatedTasks = [
      {
        task_name:
          taskType === 'causal_lm'
            ? 'task_generation'
            : taskType === 'token_classification'
            ? 'task_ner'
            : taskType === 'regression_scoring'
            ? 'task_regression'
            : 'task_classification',
        task_type: taskType === 'regression_scoring' ? 'sentence_classification' : taskType,
        strategy:
          taskType === 'causal_lm'
            ? 'CAUSAL_LM'
            : taskType === 'token_classification'
            ? 'TOKEN_CLASSIFICATION'
            : 'SENTENCE_CLASSIFICATION',
        num_classes: taskType === 'causal_lm' ? (modelConfig.vocab_size || 30522) : taskType === 'regression_scoring' ? 1 : targetNumClasses,
        max_len: modelConfig.max_seq_len || 128,
        text_column: selectedTextCol,
        label_column: selectedLabelCol,
      },
    ];
    updateModelConfig({ tasks: updatedTasks });
  };

  const handleStrategyChange = (strategy) => {
    setModelStrategy(strategy);
    if (strategy === 'scratch') {
      const arch = sizeArchetypes[selectedSizeArchetype];
      updateModelConfig({
        pretrained_backbone: 'custom_dytr',
        embed_dim: arch.d_model,
        num_layers: arch.layers,
        num_heads: arch.heads,
        max_seq_len: arch.max_seq,
      });
    } else {
      const pm = pretrainedCatalog[selectedPretrainedModel];
      updateModelConfig({
        pretrained_backbone: selectedPretrainedModel,
        custom_model_path: selectedPretrainedModel === 'custom_local' ? customModelPath : undefined,
        custom_model_format: selectedPretrainedModel === 'custom_local' ? customModelFormat : undefined,
        embed_dim: pm?.embed_dim || 768,
        num_layers: pm?.num_layers || 12,
        num_heads: pm?.num_heads || 12,
        use_task_adapters: true,
      });
    }
  };

  const handlePretrainedSelect = (modelKey) => {
    setSelectedPretrainedModel(modelKey);
    const pm = pretrainedCatalog[modelKey];
    if (pm) {
      if (modelKey === 'custom_local') {
        loadTrainedModelCheckpoint({
          backbone: 'custom_local',
          custom_model_path: customModelPath || 'checkpoint_local.safetensors',
          name: customModelPath || 'checkpoint_local.safetensors',
          format: customModelFormat,
          embed_dim: customModelParams.embed_dim,
          num_layers: customModelParams.num_layers,
          num_heads: customModelParams.num_heads,
          vocab_size: customModelParams.vocab_size,
          max_seq_len: customModelParams.max_seq_len,
        });
      } else {
        loadTrainedModelCheckpoint({
          backbone: modelKey,
          name: pm.name,
          embed_dim: pm.embed_dim,
          num_layers: pm.num_layers,
          num_heads: pm.num_heads,
        });
      }
    }
  };

  const handleSizeArchetypeChange = (archKey) => {
    setSelectedSizeArchetype(archKey);
    setIsCustomSizing(false);
    const arch = sizeArchetypes[archKey];
    if (arch) {
      updateModelConfig({
        embed_dim: arch.d_model,
        num_layers: arch.layers,
        num_heads: arch.heads,
        max_seq_len: arch.max_seq,
      });
    }
  };

  const handleCustomDimensionChange = (field, value) => {
    setIsCustomSizing(true);
    let newEmbedDim = field === 'embed_dim' ? value : (modelConfig.embed_dim || 256);
    let newHeads = field === 'num_heads' ? value : (modelConfig.num_heads || 4);
    
    // Ensure heads divides embed_dim
    if (newEmbedDim % newHeads !== 0) {
      const validHeads = [1, 2, 4, 8, 12, 16].filter((h) => newEmbedDim % h === 0);
      newHeads = validHeads[validHeads.length - 1] || 1;
    }

    updateModelConfig({
      [field]: value,
      embed_dim: newEmbedDim,
      num_heads: newHeads,
    });
  };

  // Live Python Script Synthesis
  useEffect(() => {
    const isPretrained = modelStrategy === 'pretrained';
    const isCustomLocal = isPretrained && selectedPretrainedModel === 'custom_local';
    const payload = {
      vocab_size: modelConfig.vocab_size || 30522,
      embed_dim: modelConfig.embed_dim || 256,
      num_layers: modelConfig.num_layers || 4,
      num_heads: modelConfig.num_heads || 4,
      ff_mult: modelConfig.ff_mult || 4,
      max_seq_len: modelConfig.max_seq_len || 256,
      dropout: customDropout,
      use_flash_attention: true,
      use_rotary_embedding: true,
      use_amp: true,
      use_si: false,
      use_adalora: isPretrained,
      lora_rank: loraRank,
      lora_alpha: loraAlpha,
      tasks: (modelConfig.tasks || []).map((t) => ({
        task_name: t.task_name,
        task_type: t.task_type || 'sentence_classification',
        num_classes: t.num_classes || 2,
        max_len: t.max_len || 64,
      })),
    };

    api.estimateModel(payload).then((res) => {
      if (res && res.python_code) {
        setLivePythonCode(res.python_code);
      } else {
        let fallback = '';
        if (isCustomLocal) {
          if (customArchitecturePattern === 'huggingface_standard') {
            fallback = `import torch
from transformers import AutoModel, AutoConfig
from peft import LoraConfig, get_peft_model, TaskType
from dytr import TaskHead

# 1. Load HuggingFace Standard Architecture & Weights (${customModelPath || 'checkpoint_local.safetensors'})
config = AutoConfig.from_pretrained("${customModelConfigName || 'config.json'}")
base_model = AutoModel.from_pretrained(
    "${customModelPath || './checkpoint_local'}",
    config=config,
    torch_dtype=torch.float16,
    trust_remote_code=True
)

# 2. Inject LoRA Low-Rank Adapters
peft_config = LoraConfig(
    r=${loraRank},
    lora_alpha=${loraAlpha.toFixed(1)},
    target_modules=["q_proj", "v_proj", "out_proj"],
    lora_dropout=${customDropout},
    bias="none",
    task_type=TaskType.SEQ_CLS
)
model = get_peft_model(base_model, peft_config)

# 3. Attach Task Output Head for Field: '${selectedLabelCol || 'category'}'
head = TaskHead(
    task_type="${selectedTaskType === 'regression_scoring' ? 'regression' : selectedTaskType}",
    num_classes=${selectedTaskType === 'causal_lm' ? (customModelParams.vocab_size || 30522) : selectedTaskType === 'regression_scoring' ? 1 : targetNumClasses},
    in_features=${customModelParams.embed_dim}
)
print("HuggingFace Standard Backbone Ingested (${customModelParams.num_layers} layers, ${customModelParams.embed_dim} dim) with LoRA.")`;
          } else if (customArchitecturePattern === 'pytorch_state_dict') {
            fallback = `import torch
from safetensors.torch import load_file
from dytr import DynamicTransformer, ModelConfig, TaskConfig, LoRAConfig

# 1. Instantiate Transformer Skeleton from Structural Dimensions
config = ModelConfig(
    embed_dim=${customModelParams.embed_dim},
    num_layers=${customModelParams.num_layers},
    num_heads=${customModelParams.num_heads},
    vocab_size=${customModelParams.vocab_size},
    max_seq_len=${customModelParams.max_seq_len}
)
model = DynamicTransformer(config)

# 2. Ingest Serialized State Dict Weights (${customModelPath || 'checkpoint.safetensors'})
${customModelPath?.endsWith('.safetensors') 
  ? `state_dict = load_file("${customModelPath || 'model.safetensors'}")` 
  : `state_dict = torch.load("${customModelPath || 'checkpoint.pt'}", map_location="cpu")`}
missing, unexpected = model.load_state_dict(state_dict, strict=False)
print(f"Loaded StateDict with {len(missing)} missing and {len(unexpected)} unexpected keys.")

# 3. Attach LoRA Adapters & Compile
model.inject_lora(LoRAConfig(rank=${loraRank}, alpha=${loraAlpha.toFixed(1)}, dropout=${customDropout}))
print("PyTorch State Dict Backbone Ready for Training.")`;
          } else if (customArchitecturePattern === 'onnx_runtime') {
            fallback = `import onnxruntime as ort
from dytr import ONNXBackboneWrapper, TaskHead, LoRAConfig

# 1. Initialize ONNX Runtime Inference Session
session_options = ort.SessionOptions()
session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
session = ort.InferenceSession("${customModelPath || 'model.onnx'}", session_options)

# 2. Wrap ONNX Session with Dynamic Adaptor Head
backbone = ONNXBackboneWrapper(
    session=session,
    embed_dim=${customModelParams.embed_dim},
    num_heads=${customModelParams.num_heads}
)
head = TaskHead(
    task_type="${selectedTaskType === 'regression_scoring' ? 'regression' : selectedTaskType}",
    num_classes=${selectedTaskType === 'causal_lm' ? (customModelParams.vocab_size || 30522) : selectedTaskType === 'regression_scoring' ? 1 : targetNumClasses},
    in_features=${customModelParams.embed_dim}
)
print("ONNX Runtime Execution Graph Active with ${customModelParams.embed_dim} Feature Map.")`;
          } else {
            // Default DyTR Native
            fallback = `import torch
from dytr import DynamicTransformer, ModelConfig, TaskConfig, LoRAConfig
from dytr.pretrained.loader import load_local_checkpoint

# 1. Ingest DyTR Native Local Checkpoint (${customModelPath || 'checkpoint_local.safetensors'})
${customModelConfigName ? `# Companion Config: ${customModelConfigName}` : '# Architecture: ' + customModelFamily}
backbone = load_local_checkpoint(
    checkpoint_path="${customModelPath || 'checkpoint_local.safetensors'}",
    model_family="${customModelFamily}",
    embed_dim=${customModelParams.embed_dim},
    num_layers=${customModelParams.num_layers},
    num_heads=${customModelParams.num_heads},
    vocab_size=${customModelParams.vocab_size},
    max_seq_len=${customModelParams.max_seq_len},
    lora_config=LoRAConfig(rank=${loraRank}, alpha=${loraAlpha.toFixed(1)}, dropout=${customDropout})
)

# 2. Attach Task Output Head for Target Field: '${selectedLabelCol || 'category'}'
head = backbone.create_head(
    task_type="${selectedTaskType === 'regression_scoring' ? 'regression' : selectedTaskType}",
    num_classes=${selectedTaskType === 'causal_lm' ? (customModelParams.vocab_size || 30522) : selectedTaskType === 'regression_scoring' ? 1 : targetNumClasses},
    in_features=${customModelParams.embed_dim}
)

# 3. Compile Model Ready for Adaptation
model = backbone.attach_head(head)
print("DyTR Modular Backbone Ingested and Ready for LoRA Training.")`;
          }
        } else if (isPretrained) {
          fallback = `from dytr import PretrainedBackbone, LoRAConfig, TaskHead

# 1. Load Pretrained Foundation Model with LoRA (${selectedPretrainedModel})
backbone = PretrainedBackbone.from_pretrained(
    model_name="${selectedPretrainedModel}",
    lora_config=LoRAConfig(rank=${loraRank}, alpha=${loraAlpha.toFixed(1)}, dropout=${customDropout})
)

# 2. Attach Task Output Head for Target Column: '${selectedLabelCol || 'label'}'
head = TaskHead(
    task_type="${selectedTaskType === 'regression_scoring' ? 'regression' : selectedTaskType}",
    num_classes=${selectedTaskType === 'causal_lm' ? 30522 : selectedTaskType === 'regression_scoring' ? 1 : targetNumClasses},
    in_features=backbone.hidden_size
)

# 3. Instantiate Model Ready for Training
model = backbone.attach_head(head)
print("Transfer Learning Model Compiled for Task: ${selectedTaskType}")`;
        } else {
          fallback = `from dytr import DynamicTransformer, ModelConfig, TaskConfig, TrainingStrategy

# 1. Define Native Transformer Architecture
config = ModelConfig(
    embed_dim=${modelConfig.embed_dim || 256},
    num_layers=${modelConfig.num_layers || 4},
    num_heads=${modelConfig.num_heads || 4},
    max_seq_len=${modelConfig.max_seq_len || 256},
    dropout=${customDropout},
    tasks=[
        TaskConfig(
            task_name="primary_task",
            strategy=TrainingStrategy.${
              selectedTaskType === 'causal_lm'
                ? 'CAUSAL_LM'
                : selectedTaskType === 'token_classification'
                ? 'TOKEN_CLASSIFICATION'
                : 'SENTENCE_CLASSIFICATION'
            },
            num_classes=${selectedTaskType === 'causal_lm' ? 30522 : selectedTaskType === 'regression_scoring' ? 1 : targetNumClasses},
            max_length=${modelConfig.max_seq_len || 128}
        )
    ]
)

# 2. Instantiate Neural Model
model = DynamicTransformer(config)
print("Dynamic Transformer Initialized with", sum(p.numel() for p in model.parameters()), "parameters.")`;
        }
        setLivePythonCode(fallback);
      }
    });
  }, [
    modelConfig,
    modelStrategy,
    selectedPretrainedModel,
    selectedTaskType,
    targetNumClasses,
    selectedLabelCol,
    loraRank,
    loraAlpha,
    customDropout,
    customModelPath,
    customModelFormat,
    customModelFamily,
    customModelParams,
  ]);

  const handleCopyCode = () => {
    if (livePythonCode) {
      navigator.clipboard.writeText(livePythonCode);
      setCopySuccess(true);
      setTimeout(() => setCopySuccess(false), 2000);
    }
  };

  return (
    <div className="container" style={{ maxWidth: '1000px', margin: '0 auto', padding: '2rem 1.5rem' }}>
      {/* Studio Header */}
      <div style={{ marginBottom: '2rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.5rem', flexWrap: 'wrap' }}>
          <span className="badge badge-teal" style={{ fontSize: '0.75rem', fontWeight: 800 }}>
            {d.module_builder}
          </span>
          
        </div>

        <h1 style={{ fontSize: '1.85rem', fontWeight: 800, color: 'var(--text-primary)', marginBottom: '0.35rem' }}>
          {isAr ? 'اختيار و تصميم معمارية النموذج' : 'Model Selection & Architecture Studio'}
        </h1>
        <p style={{ fontSize: '0.92rem', color: 'var(--text-secondary)', lineHeight: 1.6, margin: 0 }}>
          {isAr
            ? 'تحديد معمارية النموذج الذكي التي ستحول النصوص المدخلة إلى مخرجات ذكية، عبر 3 خطوات واضحة: تحديد نوع المخرج، اختيار مسار النموذج، وضبط الحجم المناسب لبياناتك.'
            : 'Configure the neural architecture that transforms input texts into intelligent predictions across 3 sequential decisions.'}
        </p>
      </div>

      {/* FOUNDATIONAL CONTRACT: INPUT & EXPECTED OUTPUT SPECIFICATION */}
      <div
        style={{
          background: 'var(--bg-elevated)',
          border: '1px solid var(--border-subtle)',
          borderRadius: 'var(--radius-md, 8px)',
          padding: '1.25rem',
          marginBottom: '1.75rem',
          display: 'flex',
          flexDirection: 'column',
          gap: '1rem',
          boxShadow: 'var(--shadow-sm)',
        }}
      >
        {/* Header & Source Dataset info */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.75rem', borderBottom: '1px solid var(--border-subtle)', paddingBottom: '0.85rem' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.2rem' }}>
              <span className="badge badge-teal" style={{ fontSize: '0.75rem', fontWeight: 800 }}>
                {isAr ? 'مدخلات ومخرجات البيانات' : 'Model Inputs & Targets'}
              </span>
              <strong style={{ color: 'var(--text-primary)', fontSize: '0.95rem', fontFamily: 'var(--font-mono)' }}>
                {datasetName}
              </strong>
            </div>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', margin: 0 }}>
              {isAr
                ? 'حدد العمود الذي يحتوي على النص ليتعلمه النموذج، والعمود الذي يمثل النتيجة المطلوب استنتاجها.'
                : 'Choose which column contains the text to analyze and which column represents the target output.'}
            </p>
          </div>

          <div style={{ display: 'flex', gap: '0.4rem', alignItems: 'center' }}>
            <span className="badge badge-neutral" style={{ fontSize: '0.72rem' }}>
              {datasetAnalysis.rowsCount} {isAr ? 'عينة' : 'samples'}
            </span>
            <span className="badge badge-neutral" style={{ fontSize: '0.72rem' }}>
              {datasetAnalysis.colsCount} {isAr ? 'أعمدة' : 'columns'}
            </span>
          </div>
        </div>

        {/* 4-Pillared Feature Selection & Feature Engineering Suite */}
        <FeatureEngineeringSuite
          language={language}
          datasetCols={datasetCols}
          sampleRow={sampleRow}
          datasetAnalysis={datasetAnalysis}
          selectedTextCol={selectedTextCol}
          selectedLabelCol={selectedLabelCol}
          featureMode={featureMode}
          featureColumns={featureColumns}
          featureEngineering={featureEngineering}
          onChangeConfig={handleFeatureConfigChange}
        />
      </div>

      {/* STEP 1: MODEL TAXONOMY BY OUTPUT NATURE */}
      <SectionCard
        sectionCode="03.1"
        title={
          <span style={{ display: 'inline-flex', alignItems: 'center', gap: '0.5rem' }}>
            <span>{isAr ? 'الخطوة 1: تحديد عائلة النموذج حسب نوع المخرج (Model Taxonomy)' : 'Step 1: Model Family by Output Type'}</span>
            <TermTooltip termKey="task_type" size="sm" hideLabel={true} />
          </span>
        }
        subtitle={
          isAr
            ? 'ما نوع النتيجة التي تريد من النموذج استنتاجها من النصوص؟'
            : 'What form of output do you want the model to generate from the input text?'
        }
      >
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1rem' }}>
          {modelTaxonomy.map((item) => {
            const isSelected = selectedTaskType === item.id;
            return (
              <div
                key={item.id}
                onClick={() => handleTaskTypeChange(item.id)}
                className="branch-card"
                style={{
                  cursor: 'pointer',
                  borderColor: isSelected ? 'var(--accent-teal)' : 'var(--border-subtle)',
                  background: isSelected ? 'var(--bg-elevated)' : 'var(--bg-secondary)',
                  transition: 'all 0.15s ease',
                  display: 'flex',
                  flexDirection: 'column',
                  justifyContent: 'space-between',
                  padding: '1.1rem',
                  borderRadius: 'var(--radius-sm, 6px)',
                  boxShadow: isSelected ? '0 0 0 1px var(--accent-teal)' : 'none',
                }}
              >
                <div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.45rem' }}>
                      <span style={{ color: isSelected ? 'var(--accent-teal)' : 'var(--text-primary)', fontWeight: 800, fontSize: '0.88rem' }}>
                        {isAr ? item.name_ar : item.name_en}
                      </span>
                      <TermTooltip termKey={item.tooltipKey} size="sm" hideLabel={true} />
                    </div>
                    <span className={`badge ${isSelected ? 'badge-teal' : 'badge-neutral'}`} style={{ fontSize: '0.66rem', fontWeight: 800 }}>
                      {isSelected ? (isAr ? 'المعتمد' : 'Selected') : (isAr ? 'اختيار' : 'Select')}
                    </span>
                  </div>

                  <p style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', lineHeight: 1.5, margin: '0 0 0.85rem 0' }}>
                    {isAr ? item.desc_ar : item.desc_en}
                  </p>
                </div>

                {/* Bottom Row: I/O Transformation Badge & Target Output */}
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderTop: '1px solid var(--border-subtle)', paddingTop: '0.6rem', gap: '0.5rem', flexWrap: 'wrap' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                    <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>{isAr ? 'المسار:' : 'Flow:'}</span>
                    <div
                      style={{
                        display: 'inline-flex',
                        alignItems: 'center',
                        gap: '0.35rem',
                        background: 'var(--bg-tertiary)',
                        padding: '0.18rem 0.55rem',
                        borderRadius: '4px',
                        border: '1px solid rgba(20, 184, 166, 0.25)',
                      }}
                    >
                      <span style={{ fontSize: '0.72rem', fontWeight: 700, color: 'var(--text-secondary)' }}>
                        {isAr ? item.inputLabel_ar : item.inputLabel_en}
                      </span>
                      
                      {/* Directional Crisp Vector Arrow with RTL Mirror */}
                      <svg
                        width="14"
                        height="10"
                        viewBox="0 0 14 10"
                        fill="none"
                        xmlns="http://www.w3.org/2000/svg"
                        style={{
                          color: 'var(--accent-teal)',
                          transform: isAr ? 'scaleX(-1)' : 'none',
                          flexShrink: 0,
                        }}
                      >
                        <path
                          d="M1 5H13M13 5L9 1M13 5L9 9"
                          stroke="currentColor"
                          strokeWidth="1.5"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                        />
                      </svg>

                      <span style={{ fontSize: '0.72rem', fontWeight: 800, color: 'var(--accent-teal)' }}>
                        {isAr ? item.outputLabel_ar : item.outputLabel_en}
                      </span>
                    </div>
                  </div>

                  <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
                    {isAr ? 'المخرج:' : 'Output:'} <strong style={{ color: 'var(--text-primary)' }}>{isAr ? item.outputTag_ar : item.outputTag_en}</strong>
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      </SectionCard>

      {/* STEP 2: MODEL STRATEGY (FROM SCRATCH VS PRETRAINED) */}
      <SectionCard
        sectionCode="03.2"
        title={
          <span style={{ display: 'inline-flex', alignItems: 'center', gap: '0.5rem' }}>
            <span>{isAr ? 'الخطوة 2: تحديد مسار بناء النموذج (Model Strategy)' : 'Step 2: Model Strategy'}</span>
            <TermTooltip termKey="model_strategy" size="sm" hideLabel={true} />
          </span>
        }
        subtitle={
          isAr
            ? 'كيف ستحصل على النموذج؟ هل تبني نموذجاً مخصصاً وتدربه من الصفر، أم تخصص نموذجاً لغوياً جاهزاً ومدرباً مسبقاً؟'
            : 'Choose between building a lightweight native model trained from scratch or adapting a pretrained foundation model.'
        }
      >
        {/* Pedagogical Framing Banner */}
        <div
          style={{
            padding: '0.85rem 1.1rem',
            background: 'var(--bg-secondary)',
            borderRadius: 'var(--radius-sm, 6px)',
            border: '1px solid var(--border-subtle)',
            marginBottom: '1.25rem',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.35rem' }}>
            <strong style={{ color: 'var(--accent-teal)', fontSize: '0.84rem' }}>
              {isAr ? 'المفاضلة المنهجية بين المسارين:' : 'Methodological Paradigm Comparison:'}
            </strong>
            <TermTooltip termKey="transfer_learning" size="sm" hideLabel={true} />
          </div>
          <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', lineHeight: 1.55, margin: 0 }}>
            {isAr
              ? 'يعتمد القرار على حجم البيانات المتاحة والموارد الحوسبية: التدريب من الصفر يمنحك نموذجاً فائق الخفة واستقلالية كاملة للبيانات، بينما نقل التعلم (Transfer Learning) يستعير المعرفة اللغوية الضخمة من نماذج الأساس لتعويض قلة البيانات.'
              : 'Strategy selection hinges on data scale and compute constraints: From-scratch training yields ultra-lightweight independence, while transfer learning injects massive pretrained linguistic priors for small-sample efficiency.'}
          </p>
        </div>

        {/* Primary Strategy Selection Cards */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '1.15rem', marginBottom: '1.25rem' }}>
          {/* Path A: From Scratch */}
          <div
            onClick={() => handleStrategyChange('scratch')}
            className="branch-card"
            style={{
              cursor: 'pointer',
              borderColor: modelStrategy === 'scratch' ? 'var(--accent-teal)' : 'var(--border-subtle)',
              background: modelStrategy === 'scratch' ? 'var(--bg-elevated)' : 'var(--bg-secondary)',
              transition: 'all 0.15s ease',
              borderRadius: 'var(--radius-sm, 6px)',
              padding: '1.2rem',
              display: 'flex',
              flexDirection: 'column',
              justifyContent: 'space-between',
              boxShadow: modelStrategy === 'scratch' ? '0 0 0 1px var(--accent-teal), 0 4px 12px rgba(20, 184, 166, 0.08)' : 'none',
            }}
          >
            <div>
              {/* Header & Badges */}
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '0.65rem', gap: '0.5rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.45rem', flexWrap: 'wrap' }}>
                  <span style={{ color: modelStrategy === 'scratch' ? 'var(--accent-teal)' : 'var(--text-primary)', fontWeight: 800, fontSize: '0.92rem' }}>
                    {isAr ? 'المسار أ: بناء وتدريب النموذج من الصفر (From Scratch)' : 'Path A: Native Transformer (From Scratch)'}
                  </span>
                  <TermTooltip termKey="from_scratch" size="sm" hideLabel={true} />
                </div>
                <div style={{ display: 'flex', gap: '0.35rem', flexWrap: 'wrap', justifyContent: 'flex-end' }}>
                  <span className="badge badge-neutral" style={{ fontSize: '0.65rem', fontWeight: 700 }}>
                    {isAr ? 'أصيل وخفيف' : 'Native Core'}
                  </span>
                  {modelStrategy === 'scratch' && (
                    <span className="badge badge-teal" style={{ fontSize: '0.65rem', fontWeight: 800 }}>
                      {isAr ? 'المسار المعتمد' : 'Active Choice'}
                    </span>
                  )}
                </div>
              </div>

              {/* Core Description */}
              <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', lineHeight: 1.55, margin: '0 0 0.85rem 0' }}>
                {isAr
                  ? 'نبني معمارية محول عصبي نظيفة وخفيفة، تبدأ بأوزان نقية وتتعلم بالكامل من بياناتك فقط. خفيفة جداً على الذاكرة وتضمن الخصوصية التامة.'
                  : 'Custom native transformer initialized with clean weights, trained purely on your dataset. Low memory footprint and complete data ownership.'}
              </p>

              {/* Key Quantitative Metrics Grid */}
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '0.5rem', marginBottom: '0.85rem' }}>
                <div style={{ background: 'var(--bg-tertiary)', padding: '0.45rem 0.6rem', borderRadius: '4px', border: '1px solid var(--border-subtle)' }}>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '0.25rem' }}>
                    <span style={{ fontSize: '0.66rem', color: 'var(--text-muted)' }}>{isAr ? 'حجم البيانات الموصى به:' : 'Optimal Data Scale:'}</span>
                    <TermTooltip termKey="anti_overfitting" size="xs" hideLabel={true} />
                  </div>
                  <div style={{ fontSize: '0.74rem', fontWeight: 700, color: 'var(--text-primary)', marginTop: '0.15rem' }}>{isAr ? '1,000 إلى 50,000+ عينة' : '1,000 to 50k+ rows'}</div>
                </div>

                <div style={{ background: 'var(--bg-tertiary)', padding: '0.45rem 0.6rem', borderRadius: '4px', border: '1px solid var(--border-subtle)' }}>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '0.25rem' }}>
                    <span style={{ fontSize: '0.66rem', color: 'var(--text-muted)' }}>{isAr ? 'استهلاك الذاكرة:' : 'Memory Footprint:'}</span>
                    <TermTooltip termKey="memory_footprint" size="xs" hideLabel={true} />
                  </div>
                  <div style={{ fontSize: '0.74rem', fontWeight: 700, color: 'var(--accent-teal)', marginTop: '0.15rem' }}>{isAr ? '< 50MB (خفيف جداً)' : '< 50MB (Ultra Light)'}</div>
                </div>

                <div style={{ background: 'var(--bg-tertiary)', padding: '0.45rem 0.6rem', borderRadius: '4px', border: '1px solid var(--border-subtle)' }}>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '0.25rem' }}>
                    <span style={{ fontSize: '0.66rem', color: 'var(--text-muted)' }}>{isAr ? 'سرعة التدريب:' : 'Training Velocity:'}</span>
                    <TermTooltip termKey="transformer_archetypes" size="xs" hideLabel={true} />
                  </div>
                  <div style={{ fontSize: '0.74rem', fontWeight: 700, color: 'var(--text-primary)', marginTop: '0.15rem' }}>{isAr ? 'فائقة (ثوانٍ إلى دقائق)' : 'Fast (Seconds/Mins)'}</div>
                </div>

                <div style={{ background: 'var(--bg-tertiary)', padding: '0.45rem 0.6rem', borderRadius: '4px', border: '1px solid var(--border-subtle)' }}>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '0.25rem' }}>
                    <span style={{ fontSize: '0.66rem', color: 'var(--text-muted)' }}>{isAr ? 'الخصوصية والسيادة:' : 'Privacy & Autonomy:'}</span>
                    <TermTooltip termKey="data_sovereignty" size="xs" hideLabel={true} />
                  </div>
                  <div style={{ fontSize: '0.74rem', fontWeight: 700, color: 'var(--text-primary)', marginTop: '0.15rem' }}>{isAr ? '100% محلية ومعزولة' : '100% Air-Gapped'}</div>
                </div>
              </div>
            </div>

            {/* When to Choose Box */}
            <div
              style={{
                fontSize: '0.74rem',
                color: 'var(--text-secondary)',
                borderTop: '1px solid var(--border-subtle)',
                paddingTop: '0.6rem',
                background: modelStrategy === 'scratch' ? 'rgba(20, 184, 166, 0.05)' : 'transparent',
                borderRadius: '4px',
                padding: '0.5rem',
              }}
            >
              <strong style={{ color: 'var(--accent-teal)' }}>{isAr ? 'متى تختاره؟' : 'When to choose:'}</strong>{' '}
              <span>{isAr ? 'عند الحاجة لنموذج سريع ومخصص تماماً لبياناتك ومفردات مجالك الخاص.' : 'When you need a lightweight, fast, self-contained model tailored exclusively to your domain corpus.'}</span>
            </div>
          </div>

          {/* Path B: Pretrained Foundation Transfer */}
          <div
            onClick={() => handleStrategyChange('pretrained')}
            className="branch-card"
            style={{
              cursor: 'pointer',
              borderColor: modelStrategy === 'pretrained' ? 'var(--accent-teal)' : 'var(--border-subtle)',
              background: modelStrategy === 'pretrained' ? 'var(--bg-elevated)' : 'var(--bg-secondary)',
              transition: 'all 0.15s ease',
              borderRadius: 'var(--radius-sm, 6px)',
              padding: '1.2rem',
              display: 'flex',
              flexDirection: 'column',
              justifyContent: 'space-between',
              boxShadow: modelStrategy === 'pretrained' ? '0 0 0 1px var(--accent-teal), 0 4px 12px rgba(20, 184, 166, 0.08)' : 'none',
            }}
          >
            <div>
              {/* Header & Badges */}
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '0.65rem', gap: '0.5rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.45rem', flexWrap: 'wrap' }}>
                  <span style={{ color: modelStrategy === 'pretrained' ? 'var(--accent-teal)' : 'var(--text-primary)', fontWeight: 800, fontSize: '0.92rem' }}>
                    {isAr ? 'المسار ب: تخصيص نموذج مدرب مسبقاً (Transfer Learning)' : 'Path B: Pretrained Foundation (Transfer Learning)'}
                  </span>
                  <TermTooltip termKey="foundation_model" size="sm" hideLabel={true} />
                </div>
                <div style={{ display: 'flex', gap: '0.35rem', flexWrap: 'wrap', justifyContent: 'flex-end' }}>
                  <span className="badge badge-neutral" style={{ fontSize: '0.65rem', fontWeight: 700 }}>
                    {isAr ? 'أوزان لغوية جاهزة' : 'Pretrained Weights'}
                  </span>
                  {modelStrategy === 'pretrained' && (
                    <span className="badge badge-teal" style={{ fontSize: '0.65rem', fontWeight: 800 }}>
                      {isAr ? 'المسار المعتمد' : 'Active Choice'}
                    </span>
                  )}
                </div>
              </div>

              {/* Core Description */}
              <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', lineHeight: 1.55, margin: '0 0 0.85rem 0' }}>
                {isAr
                  ? 'استغلال نموذج لغوي ذكي تم تدريبه مسبقاً على مليارات النصوص وتكييفه لبياناتك باستخدام تقنية LoRA الفعالة، مما يحقق دقة عالية حتى لو كانت البيانات قليلة.'
                  : 'Leverage deep linguistic knowledge from foundation models pretrained on massive corpora, adapted via parameter-efficient LoRA.'}
              </p>

              {/* Key Quantitative Metrics Grid */}
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '0.5rem', marginBottom: '0.85rem' }}>
                <div style={{ background: 'var(--bg-tertiary)', padding: '0.45rem 0.6rem', borderRadius: '4px', border: '1px solid var(--border-subtle)' }}>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '0.25rem' }}>
                    <span style={{ fontSize: '0.66rem', color: 'var(--text-muted)' }}>{isAr ? 'حجم البيانات الموصى به:' : 'Optimal Data Scale:'}</span>
                    <TermTooltip termKey="transfer_learning" size="xs" hideLabel={true} />
                  </div>
                  <div style={{ fontSize: '0.74rem', fontWeight: 700, color: 'var(--accent-teal)', marginTop: '0.15rem' }}>{isAr ? '100 إلى 5,000 عينة' : '100 to 5k rows'}</div>
                </div>

                <div style={{ background: 'var(--bg-tertiary)', padding: '0.45rem 0.6rem', borderRadius: '4px', border: '1px solid var(--border-subtle)' }}>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '0.25rem' }}>
                    <span style={{ fontSize: '0.66rem', color: 'var(--text-muted)' }}>{isAr ? 'استهلاك الذاكرة:' : 'Memory Footprint:'}</span>
                    <TermTooltip termKey="memory_footprint" size="xs" hideLabel={true} />
                  </div>
                  <div style={{ fontSize: '0.74rem', fontWeight: 700, color: 'var(--text-primary)', marginTop: '0.15rem' }}>{isAr ? '~500MB إلى 2GB VRAM' : '~500MB - 2GB VRAM'}</div>
                </div>

                <div style={{ background: 'var(--bg-tertiary)', padding: '0.45rem 0.6rem', borderRadius: '4px', border: '1px solid var(--border-subtle)' }}>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '0.25rem' }}>
                    <span style={{ fontSize: '0.66rem', color: 'var(--text-muted)' }}>{isAr ? 'تقنية الضبط:' : 'Tuning Mechanism:'}</span>
                    <TermTooltip termKey="lora_adapters" size="xs" hideLabel={true} />
                  </div>
                  <div style={{ fontSize: '0.74rem', fontWeight: 700, color: 'var(--accent-teal)', marginTop: '0.15rem' }}>LoRA (Rank=8)</div>
                </div>

                <div style={{ background: 'var(--bg-tertiary)', padding: '0.45rem 0.6rem', borderRadius: '4px', border: '1px solid var(--border-subtle)' }}>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '0.25rem' }}>
                    <span style={{ fontSize: '0.66rem', color: 'var(--text-muted)' }}>{isAr ? 'القوة المعرفية:' : 'Prior Knowledge:'}</span>
                    <TermTooltip termKey="foundation_model" size="xs" hideLabel={true} />
                  </div>
                  <div style={{ fontSize: '0.74rem', fontWeight: 700, color: 'var(--text-primary)', marginTop: '0.15rem' }}>{isAr ? 'مليارات النصوص العامة' : 'Billions of Tokens'}</div>
                </div>
              </div>
            </div>

            {/* When to Choose Box */}
            <div
              style={{
                fontSize: '0.74rem',
                color: 'var(--text-secondary)',
                borderTop: '1px solid var(--border-subtle)',
                paddingTop: '0.6rem',
                background: modelStrategy === 'pretrained' ? 'rgba(20, 184, 166, 0.05)' : 'transparent',
                borderRadius: '4px',
                padding: '0.5rem',
              }}
            >
              <strong style={{ color: 'var(--accent-teal)' }}>{isAr ? 'متى تختاره؟' : 'When to choose:'}</strong>{' '}
              <span>{isAr ? 'عند الرغبة في دقة لغوية قصوى وفهم سياقي عميق مع بيانات تدريب محدودة.' : 'When you want maximum language precision and contextual depth with limited training samples.'}</span>
            </div>
          </div>
        </div>

        {/* Pretrained Models Catalog & LoRA Inspector (Active on Path B) */}
        {modelStrategy === 'pretrained' && (
          <div
            style={{
              background: 'var(--bg-tertiary)',
              padding: '1.2rem',
              borderRadius: 'var(--radius-md, 8px)',
              border: '1px solid rgba(20, 184, 166, 0.3)',
              marginBottom: '1rem',
            }}
          >
            {/* Catalog Header */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.85rem', flexWrap: 'wrap', gap: '0.5rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.45rem' }}>
                <span style={{ fontSize: '0.86rem', fontWeight: 800, color: 'var(--text-primary)' }}>
                  {isAr ? 'اختر النموذج اللغوي التأسيسي المناسب لمهمتك:' : 'Select Pretrained Foundation Backbone:'}
                </span>
                <TermTooltip termKey="foundation_model" size="sm" hideLabel={true} />
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                <span className="badge badge-teal" style={{ fontSize: '0.68rem', fontFamily: 'var(--font-mono)', fontWeight: 800 }}>
                  PEFT / LoRA Adapter Active
                </span>
                <TermTooltip termKey="lora_adapters" size="xs" hideLabel={true} />
              </div>
            </div>

            {/* Models Grid */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: '0.85rem', marginBottom: '1rem' }}>
              {Object.values(pretrainedCatalog).map((pm) => {
                const isSelected = selectedPretrainedModel === pm.id;
                return (
                  <div
                    key={pm.id}
                    onClick={() => handlePretrainedSelect(pm.id)}
                    className="branch-card"
                    style={{
                      cursor: 'pointer',
                      borderColor: isSelected ? 'var(--accent-teal)' : 'var(--border-subtle)',
                      background: isSelected ? 'var(--bg-elevated)' : 'var(--bg-secondary)',
                      padding: '0.95rem',
                      borderRadius: '6px',
                      transition: 'all 0.15s ease',
                      display: 'flex',
                      flexDirection: 'column',
                      justifyContent: 'space-between',
                      boxShadow: isSelected ? '0 0 0 1px var(--accent-teal)' : 'none',
                    }}
                  >
                    <div>
                      {/* Model Title & Category */}
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '0.35rem', gap: '0.4rem' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', flexWrap: 'wrap' }}>
                          <span style={{ color: isSelected ? 'var(--accent-teal)' : 'var(--text-primary)', fontWeight: 800, fontSize: '0.86rem' }}>
                            {pm.name}
                          </span>
                          <TermTooltip termKey={pm.tooltipKey} size="xs" hideLabel={true} />
                        </div>
                        <div style={{ display: 'flex', gap: '0.25rem' }}>
                          <span className={`badge ${isSelected ? 'badge-teal' : 'badge-neutral'}`} style={{ fontSize: '0.62rem', fontWeight: 800 }}>
                            {isSelected ? (isAr ? 'المعتمد' : 'Selected') : (isAr ? 'اختيار' : 'Select')}
                          </span>
                        </div>
                      </div>

                      <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', fontWeight: 700, marginBottom: '0.35rem' }}>
                        {isAr ? pm.category_ar : pm.category_en}
                      </div>

                      {/* Crisp Description */}
                      <p style={{ fontSize: '0.76rem', color: 'var(--text-secondary)', lineHeight: 1.45, margin: '0 0 0.5rem 0' }}>
                        {isAr ? pm.desc_ar : pm.desc_en}
                      </p>
                    </div>

                    {/* Best For Tag */}
                    <div style={{ fontSize: '0.72rem', borderTop: '1px solid var(--border-subtle)', paddingTop: '0.4rem', display: 'flex', alignItems: 'center', gap: '0.3rem', flexWrap: 'wrap' }}>
                      <span style={{ color: 'var(--text-muted)' }}>{isAr ? 'الأنسب لـ:' : 'Best for:'}</span>
                      <strong style={{ color: 'var(--accent-teal)' }}>{isAr ? pm.bestFor_ar : pm.bestFor_en}</strong>
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Custom Local Model Ingestion & Parameter Tuning Drawer */}
            {selectedPretrainedModel === 'custom_local' && (
              <div
                style={{
                  marginTop: '1rem',
                  background: 'var(--bg-secondary)',
                  border: '1px solid var(--accent-teal)',
                  borderRadius: 'var(--radius-md, 8px)',
                  padding: '1.25rem',
                  boxShadow: 'var(--shadow-md)',
                }}
              >
                {/* Header */}
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem', flexWrap: 'wrap', gap: '0.5rem', borderBottom: '1px solid var(--border-subtle)', paddingBottom: '0.75rem' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.45rem' }}>
                    <span style={{ fontSize: '0.9rem', fontWeight: 800, color: 'var(--accent-teal)' }}>
                      {isAr ? 'إعداد ورفع النموذج ونقطة الحفظ المحلية (Local Checkpoint Ingestion):' : 'Local Checkpoint & Model Ingestion Suite:'}
                    </span>
                    <TermTooltip termKey="custom_local_model" size="sm" hideLabel={true} />
                  </div>
                </div>

                {uploadSuccessMessage && (
                  <div
                    style={{
                      background: 'rgba(16, 185, 129, 0.15)',
                      border: '1px solid var(--accent-emerald)',
                      color: 'var(--accent-emerald)',
                      padding: '0.65rem 0.95rem',
                      borderRadius: '6px',
                      fontSize: '0.8rem',
                      fontWeight: 700,
                      marginBottom: '1rem',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      gap: '0.5rem',
                    }}
                  >
                    <span>{uploadSuccessMessage}</span>
                    {autoDetectedFromConfig && (
                      <span className="badge badge-teal" style={{ fontSize: '0.65rem', fontWeight: 800 }}>
                        {isAr ? 'تطابق تلقائي 100%' : 'Auto-Aligned Parity'}
                      </span>
                    )}
                  </div>
                )}

                {/* Dual File Ingestion Grid: Model Weights + Companion Config JSON */}
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1rem', marginBottom: '1.25rem' }}>
                  {/* Card 1: Weights & Checkpoints (.safetensors, .pt, .bin, .onnx) */}
                  <div
                    onDragOver={(e) => {
                      e.preventDefault();
                      setIsDragOverModel(true);
                    }}
                    onDragLeave={() => setIsDragOverModel(false)}
                    onDrop={handleModelFileDrop}
                    style={{
                      border: isDragOverModel ? '2px dashed var(--accent-teal)' : '2px dashed var(--border-medium)',
                      background: isDragOverModel ? 'var(--bg-elevated)' : 'var(--bg-card)',
                      borderRadius: '8px',
                      padding: '1.25rem',
                      textAlign: 'center',
                      transition: 'all 0.2s ease',
                      cursor: 'pointer',
                      display: 'flex',
                      flexDirection: 'column',
                      justifyContent: 'space-between',
                    }}
                    onClick={() => document.getElementById('custom-model-file-input')?.click()}
                  >
                    <input
                      id="custom-model-file-input"
                      type="file"
                      multiple
                      accept=".safetensors,.pt,.pth,.bin,.onnx,.json,.zip"
                      onChange={handleModelFileChange}
                      style={{ display: 'none' }}
                    />

                    <div>
                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.35rem', marginBottom: '0.35rem' }}>
                        <span style={{ fontWeight: 800, fontSize: '0.88rem', color: 'var(--text-primary)' }}>
                          {isAr ? '1. ملف أوزان ونقاط حفظ النموذج:' : '1. Model Weights & Checkpoint:'}
                        </span>
                        <TermTooltip termKey="custom_local_model" size="xs" hideLabel={true} />
                      </div>

                      <div style={{ fontSize: '0.78rem', color: customModelFile ? 'var(--accent-teal)' : 'var(--text-secondary)', fontWeight: 700, marginBottom: '0.35rem' }}>
                        {isUploadingModel
                          ? (isAr ? 'جاري فحص وتثبيت مصفوفات النموذج...' : 'Analyzing & Ingesting Weights...')
                          : customModelFile
                          ? (isAr ? `الملف المعتمد: ${customModelFile.name}` : `Loaded: ${customModelFile.name}`)
                          : (isAr ? 'اسحب ملف الأوزان هنا (.safetensors / .pt / .onnx)' : 'Drop weights file (.safetensors, .pt, .onnx)')}
                      </div>

                      <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginBottom: '0.75rem' }}>
                        {isAr
                          ? 'SafeTensors (.safetensors), PyTorch (.pt, .bin), ONNX (.onnx)'
                          : 'SafeTensors, PyTorch (.pt/.bin), ONNX'}
                      </div>
                    </div>

                    <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '0.5rem', flexWrap: 'wrap' }}>
                      <button
                        type="button"
                        className="btn btn-secondary"
                        style={{
                          padding: '0.4rem 0.85rem',
                          fontSize: '0.76rem',
                          fontWeight: 700,
                          background: 'var(--bg-elevated)',
                          border: '1px solid var(--border-subtle)',
                        }}
                        onClick={(e) => {
                          e.stopPropagation();
                          document.getElementById('custom-model-file-input')?.click();
                        }}
                      >
                        {isAr ? 'استعراض ملف الأوزان' : 'Browse Weights'}
                      </button>

                      {customModelFile && (
                        <div style={{ display: 'inline-flex', alignItems: 'center', gap: '0.45rem', background: 'var(--bg-tertiary)', padding: '0.3rem 0.6rem', borderRadius: '4px', border: '1px solid var(--border-subtle)' }}>
                          <span style={{ fontSize: '0.72rem', color: 'var(--accent-teal)', fontFamily: 'var(--font-mono)', fontWeight: 700 }}>
                            {(customModelFile.size / (1024 * 1024)).toFixed(1)} MB
                          </span>
                          <span className="badge badge-teal" style={{ fontSize: '0.6rem', textTransform: 'uppercase' }}>
                            {customModelFormat}
                          </span>
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Card 2: Companion Config JSON (dytr_model_config.json / config.json) */}
                  <div
                    onDragOver={(e) => {
                      e.preventDefault();
                      setIsDragOverConfig(true);
                    }}
                    onDragLeave={() => setIsDragOverConfig(false)}
                    onDrop={handleConfigFileDrop}
                    style={{
                      border: isDragOverConfig
                        ? '2px dashed var(--accent-teal)'
                        : customModelConfigFile
                        ? '2px solid rgba(20, 184, 166, 0.4)'
                        : '2px dashed var(--border-medium)',
                      background: isDragOverConfig
                        ? 'var(--bg-elevated)'
                        : customModelConfigFile
                        ? 'rgba(20, 184, 166, 0.04)'
                        : 'var(--bg-card)',
                      borderRadius: '8px',
                      padding: '1.25rem',
                      textAlign: 'center',
                      transition: 'all 0.2s ease',
                      cursor: 'pointer',
                      display: 'flex',
                      flexDirection: 'column',
                      justifyContent: 'space-between',
                    }}
                    onClick={() => document.getElementById('custom-config-file-input')?.click()}
                  >
                    <input
                      id="custom-config-file-input"
                      type="file"
                      accept=".json"
                      onChange={handleConfigFileChange}
                      style={{ display: 'none' }}
                    />

                    <div>
                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.35rem', marginBottom: '0.35rem' }}>
                        <span style={{ fontWeight: 800, fontSize: '0.88rem', color: 'var(--text-primary)' }}>
                          {isAr ? '2. ملف التهيئة المعمارية المصاحب (Config JSON):' : '2. Companion Config JSON File:'}
                        </span>
                        <TermTooltip termKey="model_config_companion" size="xs" hideLabel={true} />
                      </div>

                      <div style={{ fontSize: '0.78rem', color: customModelConfigFile ? 'var(--accent-teal)' : 'var(--text-secondary)', fontWeight: 700, marginBottom: '0.35rem' }}>
                        {customModelConfigFile
                          ? (isAr ? `تم الربط مع: ${customModelConfigFile.name}` : `Linked Config: ${customModelConfigFile.name}`)
                          : (isAr ? 'اسحب ملف dytr_model_config.json أو config.json هنا' : 'Drop dytr_model_config.json or config.json')}
                      </div>

                      <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginBottom: '0.75rem' }}>
                        {isAr
                          ? 'يقوم بالتعرف التلقائي الفوري على الأبعاد، الطبقات، والمفردات'
                          : 'Auto-extracts d_model, layers, heads, and vocab parameters'}
                      </div>
                    </div>

                    <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '0.5rem', flexWrap: 'wrap' }}>
                      <button
                        type="button"
                        className="btn btn-secondary"
                        style={{
                          padding: '0.4rem 0.85rem',
                          fontSize: '0.76rem',
                          fontWeight: 700,
                          background: 'var(--bg-elevated)',
                          border: '1px solid var(--border-subtle)',
                        }}
                        onClick={(e) => {
                          e.stopPropagation();
                          document.getElementById('custom-config-file-input')?.click();
                        }}
                      >
                        {isAr ? 'استعراض ملف التهيئة (.json)' : 'Browse Config (.json)'}
                      </button>

                      {customModelConfigRaw && (
                        <button
                          type="button"
                          className="btn btn-secondary"
                          style={{
                            padding: '0.4rem 0.75rem',
                            fontSize: '0.72rem',
                            fontWeight: 700,
                            background: showConfigJsonViewer ? 'var(--accent-teal)' : 'var(--bg-tertiary)',
                            color: showConfigJsonViewer ? '#fff' : 'var(--text-primary)',
                            border: '1px solid var(--border-subtle)',
                          }}
                          onClick={(e) => {
                            e.stopPropagation();
                            setShowConfigJsonViewer(!showConfigJsonViewer);
                          }}
                        >
                          {showConfigJsonViewer
                            ? (isAr ? 'إخفاء الـ JSON' : 'Hide JSON')
                            : (isAr ? 'معاينة الـ JSON' : 'Inspect JSON')}
                        </button>
                      )}
                    </div>
                  </div>
                </div>

                {/* Collapsible Companion Config Raw JSON Inspector */}
                {showConfigJsonViewer && customModelConfigRaw && (
                  <div
                    style={{
                      background: 'var(--bg-tertiary)',
                      borderRadius: '6px',
                      padding: '1rem',
                      border: '1px solid rgba(20, 184, 166, 0.3)',
                      marginBottom: '1.25rem',
                    }}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.6rem' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                        <span style={{ fontSize: '0.8rem', fontWeight: 800, color: 'var(--accent-teal)' }}>
                          {isAr ? `محتويات ملف التهيئة المعمارية (${customModelConfigName}):` : `Parsed Metadata (${customModelConfigName}):`}
                        </span>
                        <TermTooltip termKey="model_config_companion" size="xs" hideLabel={true} />
                      </div>
                      <span className="badge badge-teal" style={{ fontSize: '0.65rem' }}>
                        JSON Schema
                      </span>
                    </div>

                    <pre
                      style={{
                        background: 'var(--bg-primary)',
                        padding: '0.85rem',
                        borderRadius: '4px',
                        border: '1px solid var(--border-subtle)',
                        color: 'var(--text-primary)',
                        fontSize: '0.74rem',
                        fontFamily: 'var(--font-mono)',
                        maxHeight: '180px',
                        overflowY: 'auto',
                        lineHeight: 1.45,
                      }}
                    >
                      {customModelConfigRaw}
                    </pre>
                  </div>
                )}

                {/* Standard Architecture Patterns & Loading Frameworks */}
                <div style={{ background: 'var(--bg-card)', padding: '1rem', borderRadius: '6px', border: '1px solid var(--border-subtle)', marginBottom: '1.25rem' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem', flexWrap: 'wrap', gap: '0.5rem' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
                      <span style={{ fontSize: '0.82rem', fontWeight: 800, color: 'var(--text-primary)' }}>
                        {isAr ? 'هيكلية وبنية بناء وتحميل النموذج القياسية:' : 'Standard Architecture Loading Schema:'}
                      </span>
                      <TermTooltip termKey="architecture_standards" size="xs" hideLabel={true} />
                    </div>
                    <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
                      {isAr ? 'تحدد طريقة استدعاء الأوزان في كود بايثون' : 'Governs deserialization & runtime framework in Python'}
                    </span>
                  </div>

                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '0.65rem' }}>
                    {[
                      {
                        id: 'dytr_native',
                        name_ar: 'محمل DyTR المعياري المدمج',
                        name_en: 'DyTR Native Modular Loader',
                        badge: 'Native DyTR',
                        desc_ar: 'التحميل الخفيف المباشر عبر dytr.pretrained.loader مع جاهزية فورية لـ LoRA.',
                        desc_en: 'Direct modular weight loader via dytr.pretrained with instant LoRA adapter binding.',
                      },
                      {
                        id: 'huggingface_standard',
                        name_ar: 'معيار HuggingFace القياسي (AutoModel)',
                        name_en: 'HuggingFace AutoModel Standard',
                        badge: 'HuggingFace / PEFT',
                        desc_ar: 'التوافق المعياري مع فئات AutoModel و AutoConfig ومكتبة PEFT الرسمية.',
                        desc_en: 'Standard AutoModel & AutoConfig integration with official HuggingFace PEFT wrapper.',
                      },
                      {
                        id: 'pytorch_state_dict',
                        name_ar: 'مصفوفة أوزان PyTorch الخام (State Dict)',
                        name_en: 'PyTorch Raw StateDict / SafeTensors',
                        badge: 'PyTorch / SafeTensors',
                        desc_ar: 'بناء هيكل المحول العصبي وتمرير مصفوفات التنسور المخزنة يدوياً.',
                        desc_en: 'DynamicTransformer instantiation with raw tensor state_dict deserialization.',
                      },
                      {
                        id: 'onnx_runtime',
                        name_ar: 'محرك تشغيل وتسريع ONNX',
                        name_en: 'ONNX Runtime Execution Graph',
                        badge: 'ONNX Graph',
                        desc_ar: 'استيراد شبكة التدفق المحسنة وتنفيذ الاستدلال فائق السرعة عبر C++ Backend.',
                        desc_en: 'Accelerated execution graph session with dynamic task adapter routing.',
                      },
                    ].map((pattern) => {
                      const isPatternActive = customArchitecturePattern === pattern.id;
                      return (
                        <div
                          key={pattern.id}
                          onClick={() => setCustomArchitecturePattern(pattern.id)}
                          style={{
                            border: isPatternActive ? '1px solid var(--accent-teal)' : '1px solid var(--border-subtle)',
                            background: isPatternActive ? 'var(--bg-elevated)' : 'var(--bg-tertiary)',
                            borderRadius: '6px',
                            padding: '0.65rem 0.8rem',
                            cursor: 'pointer',
                            transition: 'all 0.15s ease',
                            boxShadow: isPatternActive ? '0 0 0 1px var(--accent-teal)' : 'none',
                          }}
                        >
                          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.25rem' }}>
                            <span style={{ fontSize: '0.78rem', fontWeight: 800, color: isPatternActive ? 'var(--accent-teal)' : 'var(--text-primary)' }}>
                              {isAr ? pattern.name_ar : pattern.name_en}
                            </span>
                            <span className={`badge ${isPatternActive ? 'badge-teal' : 'badge-neutral'}`} style={{ fontSize: '0.58rem' }}>
                              {pattern.badge}
                            </span>
                          </div>
                          <p style={{ fontSize: '0.7rem', color: 'var(--text-secondary)', lineHeight: 1.4, margin: 0 }}>
                            {isAr ? pattern.desc_ar : pattern.desc_en}
                          </p>
                        </div>
                      );
                    })}
                  </div>
                </div>

                {/* Local Path & Model Metadata Inputs
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '1rem', marginBottom: '1.25rem' }}>
                  <div>
                    <label style={{ display: 'flex', alignItems: 'center', gap: '0.35rem', fontSize: '0.78rem', fontWeight: 800, color: 'var(--text-primary)', marginBottom: '0.35rem' }}>
                      <span>{isAr ? 'مسار الملف أو اسم النموذج المحلي:' : 'Local File Path / Identifier:'}</span>
                      <TermTooltip termKey="custom_local_model" size="xs" hideLabel={true} />
                    </label>
                    <input
                      type="text"
                      value={customModelPath}
                      onChange={(e) => setCustomModelPath(e.target.value)}
                      placeholder="e.g. checkpoints/arabic_bert_base.safetensors"
                      style={{
                        width: '100%',
                        background: 'var(--bg-card)',
                        border: '1px solid var(--border-subtle)',
                        borderRadius: '6px',
                        padding: '0.5rem 0.75rem',
                        color: 'var(--text-primary)',
                        fontSize: '0.82rem',
                        fontFamily: 'var(--font-mono)',
                      }}
                    />
                  </div>

                  <div>
                    <label style={{ display: 'flex', alignItems: 'center', gap: '0.35rem', fontSize: '0.78rem', fontWeight: 800, color: 'var(--text-primary)', marginBottom: '0.35rem' }}>
                      <span>{isAr ? 'عائلة المعمارية للنموذج المرفوع:' : 'Model Architecture Family:'}</span>
                      <TermTooltip termKey="transformer_archetypes" size="xs" hideLabel={true} />
                    </label>
                    <select
                      value={customModelFamily}
                      onChange={(e) => setCustomModelFamily(e.target.value)}
                      className="form-select"
                      style={{ fontSize: '0.82rem', padding: '0.48rem 0.65rem', background: 'var(--bg-card)' }}
                    >
                      <option value="encoder">{isAr ? 'المشفر ثنائي الاتجاه (BERT / RoBERTa / CAMeL Encoder)' : 'Bidirectional Encoder (BERT/RoBERTa)'}</option>
                      <option value="decoder">{isAr ? 'فاك التشفير التوليدي (GPT / LLaMA / Mistral Decoder)' : 'Causal Decoder (GPT/LLaMA)'}</option>
                      <option value="seq2seq">{isAr ? 'مشفر ومفكك متكامل (T5 / BART Seq2Seq)' : 'Seq2Seq Encoder-Decoder (T5/BART)'}</option>
                      <option value="custom">{isAr ? 'معمارية DyTR مخصصة (Custom Transformer)' : 'Custom DyTR Transformer'}</option>
                    </select>
                  </div>
                </div> */}

                {/* Architecture Dimensions Sub-grid */}
                <div style={{ background: 'var(--bg-card)', padding: '1rem', borderRadius: '6px', border: '1px solid var(--border-subtle)', marginBottom: '1.25rem' }}>
                  <div style={{ fontSize: '0.8rem', fontWeight: 800, color: 'var(--text-primary)', marginBottom: '0.75rem', display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
                    <span>{isAr ? 'أبعاد ومحددات المعمارية للنموذج المحلي:' : 'Local Model Structural Dimensions:'}</span>
                    <TermTooltip termKey="embedding_dimension" size="xs" hideLabel={true} />
                  </div>

                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: '0.75rem' }}>
                    <div>
                      <label style={{ fontSize: '0.72rem', color: 'var(--text-muted)', display: 'block', marginBottom: '0.2rem' }}>
                        {isAr ? 'أبعاد التضمين (d_model):' : 'Embed Dim (d_model):'}
                      </label>
                      <input
                        type="number"
                        min="64"
                        max="4096"
                        step="64"
                        value={customModelParams.embed_dim}
                        onChange={(e) => {
                          const val = Number(e.target.value);
                          setCustomModelParams({ ...customModelParams, embed_dim: val });
                          updateModelConfig({ embed_dim: val });
                        }}
                        style={{ width: '100%', background: 'var(--bg-input)', border: '1px solid var(--border-subtle)', borderRadius: '4px', padding: '0.4rem', color: 'var(--text-primary)', fontSize: '0.8rem', fontFamily: 'var(--font-mono)' }}
                      />
                    </div>

                    <div>
                      <label style={{ fontSize: '0.72rem', color: 'var(--text-muted)', display: 'block', marginBottom: '0.2rem' }}>
                        {isAr ? 'عدد الطبقات (Layers):' : 'Num Layers:'}
                      </label>
                      <input
                        type="number"
                        min="1"
                        max="64"
                        value={customModelParams.num_layers}
                        onChange={(e) => {
                          const val = Number(e.target.value);
                          setCustomModelParams({ ...customModelParams, num_layers: val });
                          updateModelConfig({ num_layers: val });
                        }}
                        style={{ width: '100%', background: 'var(--bg-input)', border: '1px solid var(--border-subtle)', borderRadius: '4px', padding: '0.4rem', color: 'var(--text-primary)', fontSize: '0.8rem', fontFamily: 'var(--font-mono)' }}
                      />
                    </div>

                    <div>
                      <label style={{ fontSize: '0.72rem', color: 'var(--text-muted)', display: 'block', marginBottom: '0.2rem' }}>
                        {isAr ? 'رؤوس الانتباه (Heads):' : 'Num Heads:'}
                      </label>
                      <input
                        type="number"
                        min="1"
                        max="32"
                        value={customModelParams.num_heads}
                        onChange={(e) => {
                          const val = Number(e.target.value);
                          setCustomModelParams({ ...customModelParams, num_heads: val });
                          updateModelConfig({ num_heads: val });
                        }}
                        style={{ width: '100%', background: 'var(--bg-input)', border: '1px solid var(--border-subtle)', borderRadius: '4px', padding: '0.4rem', color: 'var(--text-primary)', fontSize: '0.8rem', fontFamily: 'var(--font-mono)' }}
                      />
                    </div>

                    <div>
                      <label style={{ fontSize: '0.72rem', color: 'var(--text-muted)', display: 'block', marginBottom: '0.2rem' }}>
                        {isAr ? 'حجم المفردات (Vocab):' : 'Vocab Size:'}
                      </label>
                      <input
                        type="number"
                        min="1000"
                        max="256000"
                        step="1000"
                        value={customModelParams.vocab_size}
                        onChange={(e) => {
                          const val = Number(e.target.value);
                          setCustomModelParams({ ...customModelParams, vocab_size: val });
                          updateModelConfig({ vocab_size: val });
                        }}
                        style={{ width: '100%', background: 'var(--bg-input)', border: '1px solid var(--border-subtle)', borderRadius: '4px', padding: '0.4rem', color: 'var(--text-primary)', fontSize: '0.8rem', fontFamily: 'var(--font-mono)' }}
                      />
                    </div>

                    <div>
                      <label style={{ fontSize: '0.72rem', color: 'var(--text-muted)', display: 'block', marginBottom: '0.2rem' }}>
                        {isAr ? 'أقصى سياق (Max Seq):' : 'Max Seq Length:'}
                      </label>
                      <input
                        type="number"
                        min="32"
                        max="4096"
                        step="32"
                        value={customModelParams.max_seq_len}
                        onChange={(e) => {
                          const val = Number(e.target.value);
                          setCustomModelParams({ ...customModelParams, max_seq_len: val });
                          updateModelConfig({ max_seq_len: val });
                        }}
                        style={{ width: '100%', background: 'var(--bg-input)', border: '1px solid var(--border-subtle)', borderRadius: '4px', padding: '0.4rem', color: 'var(--text-primary)', fontSize: '0.8rem', fontFamily: 'var(--font-mono)' }}
                      />
                    </div>
                  </div>
                </div>

                {/* Real-Time Parameter & Memory Footprint Card */}
                {(() => {
                  const d = customModelParams.embed_dim || 768;
                  const l = customModelParams.num_layers || 12;
                  const v = customModelParams.vocab_size || 30522;
                  const r = loraRank || 8;
                  const numClasses = selectedTaskType === 'causal_lm' ? v : selectedTaskType === 'regression_scoring' ? 1 : targetNumClasses;

                  const embParams = v * d;
                  const layerParams = l * (12 * d * d + 4 * d);
                  const headParams = d * numClasses + numClasses;
                  const loraParams = l * (4 * d * r);
                  const totalDense = embParams + layerParams + headParams;
                  const totalTrainableLoRA = loraParams + headParams;
                  const sizeMbFp16 = ((totalDense * 2) / (1024 * 1024)).toFixed(1);
                  const loraPercent = ((totalTrainableLoRA / totalDense) * 100).toFixed(2);

                  return (
                    <div style={{ background: 'var(--bg-elevated)', borderRadius: '6px', padding: '0.85rem 1rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.75rem', border: '1px solid var(--border-subtle)' }}>
                      <div>
                        <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
                          {isAr ? 'إجمالي معلمات النموذج المرفوع:' : 'Total Ingested Parameters:'}
                        </div>
                        <div style={{ fontSize: '1.1rem', fontWeight: 800, color: 'var(--text-primary)', fontFamily: 'var(--font-mono)' }}>
                          {(totalDense / 1e6).toFixed(1)}M ({totalDense.toLocaleString()})
                        </div>
                      </div>

                      <div>
                        <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
                          {isAr ? 'حجم الذاكرة (FP16):' : 'Estimated FP16 Size:'}
                        </div>
                        <div style={{ fontSize: '1.1rem', fontWeight: 800, color: 'var(--accent-teal)', fontFamily: 'var(--font-mono)' }}>
                          ~{sizeMbFp16} MB
                        </div>
                      </div>

                      <div>
                        <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
                          {isAr ? 'معلمات LoRA القابلة للتدريب:' : 'LoRA Trainable Params:'}
                        </div>
                        <div style={{ fontSize: '1.1rem', fontWeight: 800, color: 'var(--accent-emerald)', fontFamily: 'var(--font-mono)' }}>
                          {totalTrainableLoRA.toLocaleString()} ({loraPercent}%)
                        </div>
                      </div>

                      <div style={{ display: 'flex', alignItems: 'center' }}>
                        <span className="badge badge-teal" style={{ fontSize: '0.72rem', padding: '0.35rem 0.75rem', fontWeight: 800 }}>
                          {isAr ? 'جاهز للتكييف بالـ LoRA' : 'Ready for LoRA Adaptation'}
                        </span>
                      </div>
                    </div>
                  );
                })()}
              </div>
            )}
          </div>
        )}

        {/* Side-by-Side Trade-off Comparison Matrix */}
        <div
          style={{
            background: 'var(--bg-secondary)',
            borderRadius: 'var(--radius-sm, 6px)',
            border: '1px solid var(--border-subtle)',
            padding: '0.9rem 1.1rem',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.45rem', marginBottom: '0.6rem' }}>
            <span style={{ fontSize: '0.78rem', fontWeight: 800, color: 'var(--text-primary)' }}>
              {isAr ? 'مصفوفة المقارنة المعمارية السريعة:' : 'Architectural Trade-off Matrix:'}
            </span>
            <TermTooltip termKey="model_strategy" size="xs" hideLabel={true} />
          </div>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.76rem', textAlign: isAr ? 'right' : 'left' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--border-subtle)', color: 'var(--text-muted)' }}>
                  <th style={{ padding: '0.4rem 0.5rem' }}>{isAr ? 'وجه المقارنة' : 'Dimension'}</th>
                  <th style={{ padding: '0.4rem 0.5rem', color: 'var(--accent-teal)' }}>{isAr ? 'المسار أ (From Scratch)' : 'Path A (From Scratch)'}</th>
                  <th style={{ padding: '0.4rem 0.5rem', color: 'var(--accent-teal)' }}>{isAr ? 'المسار ب (Transfer Learning)' : 'Path B (Transfer Learning)'}</th>
                </tr>
              </thead>
              <tbody>
                <tr style={{ borderBottom: '1px solid var(--border-subtle)' }}>
                  <td style={{ padding: '0.45rem 0.5rem', fontWeight: 700, color: 'var(--text-secondary)' }}>
                    <div style={{ display: 'inline-flex', alignItems: 'center', gap: '0.35rem' }}>
                      <span>{isAr ? 'متطلبات حجم البيانات' : 'Data Requirement'}</span>
                      <TermTooltip termKey="anti_overfitting" size="xs" hideLabel={true} />
                    </div>
                  </td>
                  <td style={{ padding: '0.45rem 0.5rem' }}>{isAr ? 'يحتاج عينات كافية لتكوين المفردات' : 'Requires sufficient data to generalize'}</td>
                  <td style={{ padding: '0.45rem 0.5rem', fontWeight: 700, color: 'var(--accent-teal)' }}>{isAr ? 'يكتفي بعينات قليلة ومحدودة' : 'High accuracy on few samples'}</td>
                </tr>
                <tr style={{ borderBottom: '1px solid var(--border-subtle)' }}>
                  <td style={{ padding: '0.45rem 0.5rem', fontWeight: 700, color: 'var(--text-secondary)' }}>
                    <div style={{ display: 'inline-flex', alignItems: 'center', gap: '0.35rem' }}>
                      <span>{isAr ? 'استهلاك الذاكرة والعتاد' : 'Compute & Memory'}</span>
                      <TermTooltip termKey="memory_footprint" size="xs" hideLabel={true} />
                    </div>
                  </td>
                  <td style={{ padding: '0.45rem 0.5rem', fontWeight: 700, color: 'var(--accent-teal)' }}>{isAr ? 'خفيف جداً ويعمل على أجهزة عادية' : 'Ultra-lightweight on CPU/Edge'}</td>
                  <td style={{ padding: '0.45rem 0.5rem' }}>{isAr ? 'يتطلب ذاكرة أكبر للنموذج الأساسي' : 'Requires GPU memory for backbone'}</td>
                </tr>
                <tr>
                  <td style={{ padding: '0.45rem 0.5rem', fontWeight: 700, color: 'var(--text-secondary)' }}>
                    <div style={{ display: 'inline-flex', alignItems: 'center', gap: '0.35rem' }}>
                      <span>{isAr ? 'خصوصية وسرعة الاستنتاج' : 'Privacy & Latency'}</span>
                      <TermTooltip termKey="data_sovereignty" size="xs" hideLabel={true} />
                    </div>
                  </td>
                  <td style={{ padding: '0.45rem 0.5rem', fontWeight: 700, color: 'var(--accent-teal)' }}>{isAr ? 'استجابة فائقة السرعة واستقلالية 100%' : 'Near-zero latency & local sovereignty'}</td>
                  <td style={{ padding: '0.45rem 0.5rem' }}>{isAr ? 'استجابة معتمدة على حجم نموذج الأساس' : 'Latency dependent on backbone size'}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </SectionCard>

      {/* STEP 3: MODEL SIZING BASED ON DATASET SCALE */}
      <SectionCard
        sectionCode="03.3"
        title={
          <span style={{ display: 'inline-flex', alignItems: 'center', gap: '0.5rem' }}>
            <span>{isAr ? 'الخطوة 3: تحديد حجم وسعة النموذج (Model Sizing)' : 'Step 3: Model Sizing & Capacity'}</span>
            <TermTooltip termKey="transformer_archetypes" size="sm" hideLabel={true} />
          </span>
        }
        subtitle={
          isAr
            ? 'ما حجم المعمارية المناسب لحجم بياناتك وموارد جهازك لتفادي الحفظ الصم وضعف التعميم؟'
            : 'What model capacity matches your dataset volume to prevent memorization and ensure generalization?'
        }
      >
        {modelStrategy === 'scratch' ? (
          <div>
            {/* View Mode Toggle: Standard Archetypes vs Custom Fine-Tuning */}
            <div
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                marginBottom: '1rem',
                flexWrap: 'wrap',
                gap: '0.5rem',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.45rem' }}>
                <span style={{ fontSize: '0.8rem', fontWeight: 800, color: 'var(--text-primary)' }}>
                  {isAr ? 'نمط تحديد الأبعاد:' : 'Sizing Mode:'}
                </span>
                <TermTooltip termKey="anti_overfitting" size="xs" hideLabel={true} />
              </div>

              <div style={{ display: 'flex', gap: '0.35rem', background: 'var(--bg-secondary)', padding: '0.25rem', borderRadius: '6px' }}>
                <button
                  type="button"
                  onClick={() => setIsCustomSizing(false)}
                  className={`btn btn-sm ${!isCustomSizing ? 'btn-primary' : 'btn-ghost'}`}
                  style={{ fontSize: '0.72rem', padding: '0.25rem 0.65rem' }}
                >
                  {isAr ? 'القوالب القياسية الموصى بها' : 'Standard Archetypes'}
                </button>
                <button
                  type="button"
                  onClick={() => setIsCustomSizing(true)}
                  className={`btn btn-sm ${isCustomSizing ? 'btn-primary' : 'btn-ghost'}`}
                  style={{ fontSize: '0.72rem', padding: '0.25rem 0.65rem' }}
                >
                  {isAr ? 'تخصيص يدوي دقيق' : 'Custom Fine-Tuning'}
                </button>
              </div>
            </div>

            {/* If Standard Archetypes */}
            {!isCustomSizing && (
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1rem', marginBottom: '1.25rem' }}>
                {Object.values(sizeArchetypes).map((arch) => {
                  const isSelected = selectedSizeArchetype === arch.id;
                  const isRecommended = datasetAnalysis.recommendedSize === arch.id;

                  return (
                    <div
                      key={arch.id}
                      onClick={() => handleSizeArchetypeChange(arch.id)}
                      className="branch-card"
                      style={{
                        cursor: 'pointer',
                        borderColor: isSelected ? 'var(--accent-teal)' : 'var(--border-subtle)',
                        background: isSelected ? 'var(--bg-elevated)' : 'var(--bg-secondary)',
                        transition: 'all 0.15s ease',
                        position: 'relative',
                        padding: '1rem',
                        borderRadius: 'var(--radius-md, 8px)',
                        boxShadow: isSelected ? '0 0 0 1px var(--accent-teal)' : 'none',
                        display: 'flex',
                        flexDirection: 'column',
                        justifyContent: 'space-between',
                      }}
                    >
                      <div>
                        {/* Header: Title + Badges */}
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '0.4rem', gap: '0.5rem' }}>
                          <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', flexWrap: 'wrap' }}>
                            <span style={{ color: isSelected ? 'var(--accent-teal)' : 'var(--text-primary)', fontWeight: 800, fontSize: '0.9rem' }}>
                              {isAr ? arch.name_ar : arch.name_en}
                            </span>
                            {isRecommended && (
                              <span className="badge badge-teal" style={{ fontSize: '0.62rem', fontWeight: 800 }}>
                                {isAr ? 'الأنسب لبياناتك' : 'Best Match'}
                              </span>
                            )}
                          </div>
                          <span className={`badge ${isSelected ? 'badge-teal' : 'badge-neutral'}`} style={{ fontSize: '0.65rem', fontWeight: 700 }}>
                            {isSelected ? (isAr ? 'المعتمد' : 'Selected') : (isAr ? 'اختيار' : 'Select')}
                          </span>
                        </div>

                        {/* Short Clear Description */}
                        <p style={{ fontSize: '0.76rem', color: 'var(--text-secondary)', lineHeight: 1.5, margin: '0.3rem 0 0.6rem 0' }}>
                          {isAr ? arch.desc_ar : arch.desc_en}
                        </p>
                      </div>

                      {/* Specs Row */}
                      <div
                        style={{
                          background: 'var(--bg-tertiary)',
                          borderRadius: '6px',
                          padding: '0.45rem 0.65rem',
                          border: '1px solid var(--border-subtle)',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'space-between',
                          flexWrap: 'wrap',
                          gap: '0.4rem',
                          fontSize: '0.72rem',
                          fontFamily: 'var(--font-mono)',
                          color: 'var(--text-secondary)',
                        }}
                      >
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                          <strong style={{ color: isSelected ? 'var(--accent-teal)' : 'var(--text-primary)' }}>d_model: {arch.d_model}</strong>
                          <span>•</span>
                          <span>{arch.layers} {isAr ? 'طبقات' : 'Layers'}</span>
                        </div>
                        <span style={{ color: 'var(--text-muted)', fontSize: '0.68rem' }}>
                          {isAr ? arch.suitableFor_ar : arch.suitableFor_en}
                        </span>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}

            {/* If Custom Fine-Tuning */}
            {isCustomSizing && (
              <div
                style={{
                  background: 'var(--bg-elevated)',
                  borderRadius: '6px',
                  border: '1px solid var(--accent-teal)',
                  padding: '1rem',
                  marginBottom: '1.25rem',
                }}
              >
                <div style={{ fontSize: '0.8rem', fontWeight: 800, color: 'var(--accent-teal)', marginBottom: '0.75rem', display: 'flex', alignItems: 'center', gap: '0.45rem' }}>
                  <span>{isAr ? 'لوحة التخصيص المعماري الدقيق (Custom Transformer Dimensions):' : 'Custom Architecture Hyperparameter Tuning:'}</span>
                  <TermTooltip termKey="transformer_archetypes" size="xs" hideLabel={true} />
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '0.85rem' }}>
                  {/* d_model */}
                  <div>
                    <div style={{ fontSize: '0.74rem', fontWeight: 700, color: 'var(--text-secondary)', marginBottom: '0.35rem', display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
                      <span>{isAr ? 'البُعد الخفي (d_model):' : 'Hidden Dim (d_model):'}</span>
                      <TermTooltip termKey="d_model" size="xs" hideLabel={true} />
                    </div>
                    <div style={{ display: 'flex', gap: '0.3rem', flexWrap: 'wrap' }}>
                      {[64, 128, 256, 512, 768].map((val) => (
                        <button
                          key={val}
                          type="button"
                          onClick={() => handleCustomDimensionChange('embed_dim', val)}
                          className={`btn btn-sm ${(modelConfig.embed_dim || 256) === val ? 'btn-primary' : 'btn-secondary'}`}
                          style={{ fontSize: '0.7rem', padding: '0.2rem 0.5rem', fontFamily: 'var(--font-mono)' }}
                        >
                          {val}
                        </button>
                      ))}
                    </div>
                  </div>

                  {/* num_layers */}
                  <div>
                    <div style={{ fontSize: '0.74rem', fontWeight: 700, color: 'var(--text-secondary)', marginBottom: '0.35rem', display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
                      <span>{isAr ? 'عدد الطبقات (Layers):' : 'Layers (Depth):'}</span>
                      <TermTooltip termKey="transformer_layers" size="xs" hideLabel={true} />
                    </div>
                    <div style={{ display: 'flex', gap: '0.3rem', flexWrap: 'wrap' }}>
                      {[1, 2, 4, 6, 8, 12].map((val) => (
                        <button
                          key={val}
                          type="button"
                          onClick={() => handleCustomDimensionChange('num_layers', val)}
                          className={`btn btn-sm ${(modelConfig.num_layers || 4) === val ? 'btn-primary' : 'btn-secondary'}`}
                          style={{ fontSize: '0.7rem', padding: '0.2rem 0.5rem', fontFamily: 'var(--font-mono)' }}
                        >
                          {val}L
                        </button>
                      ))}
                    </div>
                  </div>

                  {/* num_heads */}
                  <div>
                    <div style={{ fontSize: '0.74rem', fontWeight: 700, color: 'var(--text-secondary)', marginBottom: '0.35rem', display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
                      <span>{isAr ? 'رؤوس الانتباه (Heads):' : 'Attention Heads:'}</span>
                      <TermTooltip termKey="attention_heads" size="xs" hideLabel={true} />
                    </div>
                    <div style={{ display: 'flex', gap: '0.3rem', flexWrap: 'wrap' }}>
                      {[1, 2, 4, 8, 12, 16].map((val) => {
                        const isDivisible = (modelConfig.embed_dim || 256) % val === 0;
                        return (
                          <button
                            key={val}
                            type="button"
                            disabled={!isDivisible}
                            onClick={() => handleCustomDimensionChange('num_heads', val)}
                            className={`btn btn-sm ${(modelConfig.num_heads || 4) === val ? 'btn-primary' : 'btn-secondary'}`}
                            style={{
                              fontSize: '0.7rem',
                              padding: '0.2rem 0.5rem',
                              fontFamily: 'var(--font-mono)',
                              opacity: isDivisible ? 1 : 0.4,
                            }}
                          >
                            {val}H
                          </button>
                        );
                      })}
                    </div>
                  </div>

                  {/* Dropout */}
                  <div>
                    <div style={{ fontSize: '0.74rem', fontWeight: 700, color: 'var(--text-secondary)', marginBottom: '0.35rem', display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
                      <span>{isAr ? 'نسبة الإسقاط (Dropout):' : 'Dropout Regularization:'}</span>
                      <TermTooltip termKey="anti_overfitting" size="xs" hideLabel={true} />
                    </div>
                    <div style={{ display: 'flex', gap: '0.3rem', flexWrap: 'wrap' }}>
                      {[0.0, 0.05, 0.1, 0.2].map((val) => (
                        <button
                          key={val}
                          type="button"
                          onClick={() => setCustomDropout(val)}
                          className={`btn btn-sm ${customDropout === val ? 'btn-primary' : 'btn-secondary'}`}
                          style={{ fontSize: '0.7rem', padding: '0.2rem 0.5rem', fontFamily: 'var(--font-mono)' }}
                        >
                          {(val * 100).toFixed(0)}%
                        </button>
                      ))}
                    </div>
                  </div>
                </div>

                <div style={{ marginTop: '0.65rem', fontSize: '0.72rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                  {isAr
                    ? `* البُعد الفردي لكل رأس انتباه: d_head = ${Math.floor((modelConfig.embed_dim || 256) / (modelConfig.num_heads || 4))} | التغذية الأمامية: d_ffn = ${(modelConfig.embed_dim || 256) * 4}`
                    : `* Subspace head dimension: d_head = ${Math.floor((modelConfig.embed_dim || 256) / (modelConfig.num_heads || 4))} | Feed-forward: d_ffn = ${(modelConfig.embed_dim || 256) * 4}`}
                </div>
              </div>
            )}
          </div>
        ) : (
          /* Transfer Learning Complete LoRA Suite (Presets + Granular Tuning) */
          <div style={{ marginBottom: '1.25rem' }}>
            {/* Dynamic Foundation Backbone Specification Card */}
            <div
              style={{
                background: 'var(--bg-secondary)',
                borderRadius: '6px',
                border: '1px solid var(--border-subtle)',
                padding: '0.85rem 1rem',
                marginBottom: '1rem',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                flexWrap: 'wrap',
                gap: '0.75rem',
              }}
            >
              <div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.45rem', marginBottom: '0.25rem' }}>
                  <span style={{ fontWeight: 800, fontSize: '0.86rem', color: 'var(--text-primary)' }}>
                    {pretrainedCatalog[selectedPretrainedModel]?.name || 'Pre-trained Backbone'}
                  </span>
                  <span className="badge badge-teal" style={{ fontSize: '0.66rem', fontWeight: 800 }}>
                    {isAr ? 'أوزان الأساس مجمدة (Frozen Backbone)' : 'Base Frozen'}
                  </span>
                  <TermTooltip termKey={pretrainedCatalog[selectedPretrainedModel]?.tooltipKey || 'backbone_bert'} size="xs" hideLabel={true} />
                </div>
                <div style={{ fontSize: '0.74rem', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
                  {isAr
                    ? `النموذج الأساسي جاهز وثابت للقراءة فقط (Read-Only)، وسنضيف عليه فقط طبقة تكييف ذكية وخفيفة (LoRA) لتتعلم من بياناتك دون استهلاك الذاكرة.`
                    : `The base backbone is loaded in read-only mode, and we attach a lightweight LoRA adapter to learn from your data with minimal memory.`}
                </div>
              </div>

              <div style={{ display: 'flex', gap: '0.35rem', fontFamily: 'var(--font-mono)', fontSize: '0.72rem', flexWrap: 'wrap' }}>
                <span className="badge badge-neutral">{isAr ? `${pretrainedCatalog[selectedPretrainedModel]?.num_layers || 12} طبقات فهم` : `${pretrainedCatalog[selectedPretrainedModel]?.num_layers || 12} Layers`}</span>
                <span className="badge badge-neutral">{isAr ? `${pretrainedCatalog[selectedPretrainedModel]?.embed_dim || 768} بُعد للكلمة` : `d_model: ${pretrainedCatalog[selectedPretrainedModel]?.embed_dim || 768}`}</span>
                <span className="badge badge-neutral">{isAr ? `${pretrainedCatalog[selectedPretrainedModel]?.num_heads || 12} عدسات انتباه` : `${pretrainedCatalog[selectedPretrainedModel]?.num_heads || 12} Heads`}</span>
              </div>
            </div>

            {/* Scientific LoRA Presets Header & Cards */}
            <div style={{ marginBottom: '1rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.6rem', flexWrap: 'wrap', gap: '0.4rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                  <span style={{ fontSize: '0.78rem', fontWeight: 800, color: 'var(--text-primary)' }}>
                    {isAr ? 'المعايير المعتمدة في أبحاث الذكاء الاصطناعي (Scientific Presets):' : 'Established AI Research Benchmarks:'}
                  </span>
                  <TermTooltip termKey="lora_config" size="xs" hideLabel={true} />
                </div>
                <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
                  {isAr ? 'انقر على أي قالب لتطبيقه مباشرة' : 'Click any preset to apply instantly'}
                </span>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: '0.85rem' }}>
                {loraPresets.map((preset) => {
                  const isPresetActive = loraRank === preset.rank && loraAlpha === preset.alpha;
                  return (
                    <div
                      key={preset.id}
                      onClick={() => {
                        setLoraRank(preset.rank);
                        setLoraAlpha(preset.alpha);
                      }}
                      className="branch-card"
                      style={{
                        cursor: 'pointer',
                        padding: '0.85rem 1rem',
                        borderColor: isPresetActive ? 'var(--accent-teal)' : 'var(--border-subtle)',
                        background: isPresetActive ? 'var(--bg-elevated)' : 'var(--bg-secondary)',
                        borderRadius: '6px',
                        boxShadow: isPresetActive ? '0 0 0 1px var(--accent-teal)' : 'none',
                        transition: 'all 0.15s ease',
                        display: 'flex',
                        flexDirection: 'column',
                        justifyContent: 'space-between',
                      }}
                    >
                      <div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.35rem' }}>
                          <span style={{ fontSize: '0.8rem', fontWeight: 800, color: isPresetActive ? 'var(--accent-teal)' : 'var(--text-primary)' }}>
                            {isAr ? preset.name_ar : preset.name_en}
                          </span>
                          <span className={`badge ${isPresetActive ? 'badge-teal' : 'badge-neutral'}`} style={{ fontSize: '0.64rem', fontFamily: 'var(--font-mono)' }}>
                            r = {preset.rank}
                          </span>
                        </div>
                        <p style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', margin: '0 0 0.5rem 0', lineHeight: 1.45 }}>
                          {isAr ? preset.desc_ar : preset.desc_en}
                        </p>
                      </div>

                      <div style={{ fontSize: '0.68rem', fontFamily: 'var(--font-mono)', borderTop: '1px solid var(--border-subtle)', paddingTop: '0.45rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <span style={{ color: 'var(--accent-teal)', fontWeight: 700 }}>α = {preset.alpha} (2.0x)</span>
                        <span style={{ color: 'var(--accent-emerald)', fontWeight: 700 }}>
                          {isAr ? `معلمات: ${preset.trainableStr}` : `Trainable: ${preset.trainableStr}`}
                        </span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Granular LoRA Hyperparameter Tuning Suite */}
            <div
              style={{
                background: 'var(--bg-elevated)',
                borderRadius: '6px',
                border: '1px solid var(--accent-teal)',
                padding: '1rem',
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.85rem', flexWrap: 'wrap', gap: '0.5rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.45rem' }}>
                  <span style={{ fontSize: '0.82rem', fontWeight: 800, color: 'var(--accent-teal)' }}>
                    {isAr ? 'تعديل وتخصيص المعاملات يدوياً (Manual LoRA Tuning):' : 'Manual LoRA Tuning:'}
                  </span>
                  <TermTooltip termKey="lora_config" size="xs" hideLabel={true} />
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem', flexWrap: 'wrap' }}>
                  <span className="badge badge-teal" style={{ fontSize: '0.68rem', fontFamily: 'var(--font-mono)' }}>
                    Scaling Factor (α/r): {(loraAlpha / loraRank).toFixed(2)}x
                  </span>
                  <span className="badge badge-emerald" style={{ fontSize: '0.68rem', fontFamily: 'var(--font-mono)' }}>
                    Trainable: &lt; {((36864 * loraRank) / 1.1e6).toFixed(2)}%
                  </span>
                  <TermTooltip termKey="trainable_parameters" size="xs" hideLabel={true} />
                </div>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '0.85rem', background: 'var(--bg-secondary)', padding: '0.75rem', borderRadius: '6px' }}>
                {/* LoRA Rank r */}
                <div>
                  <div style={{ fontSize: '0.74rem', fontWeight: 700, color: 'var(--text-secondary)', marginBottom: '0.35rem', display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
                    <span>{isAr ? 'سعة التكييف (Rank r):' : 'LoRA Rank (r):'}</span>
                    <TermTooltip termKey="lora_rank" size="xs" hideLabel={true} />
                  </div>
                  <div style={{ display: 'flex', gap: '0.3rem', flexWrap: 'wrap' }}>
                    {[4, 8, 16, 32, 64].map((rVal) => (
                      <button
                        key={rVal}
                        type="button"
                        onClick={() => {
                          setLoraRank(rVal);
                          setLoraAlpha(rVal * 2);
                        }}
                        className={`btn btn-sm ${loraRank === rVal ? 'btn-primary' : 'btn-ghost'}`}
                        style={{ fontSize: '0.7rem', padding: '0.2rem 0.5rem', fontFamily: 'var(--font-mono)' }}
                      >
                        r = {rVal}
                      </button>
                    ))}
                  </div>
                  <div style={{ fontSize: '0.66rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
                    {loraRank === 8 ? (isAr ? 'المعيار القياسي لتصنيف النصوص' : 'Standard for classification') : (isAr ? `سعة الفضاء: ${loraRank} أبعاد` : `Subspace: ${loraRank}d`)}
                  </div>
                </div>

                {/* LoRA Alpha */}
                <div>
                  <div style={{ fontSize: '0.74rem', fontWeight: 700, color: 'var(--text-secondary)', marginBottom: '0.35rem', display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
                    <span>{isAr ? 'قوة التأثير (Scaling Alpha α):' : 'Scaling Alpha (α):'}</span>
                    <TermTooltip termKey="lora_alpha" size="xs" hideLabel={true} />
                  </div>
                  <div style={{ display: 'flex', gap: '0.3rem', flexWrap: 'wrap' }}>
                    {[8, 16, 32, 64, 128].map((aVal) => (
                      <button
                        key={aVal}
                        type="button"
                        onClick={() => setLoraAlpha(aVal)}
                        className={`btn btn-sm ${loraAlpha === aVal ? 'btn-primary' : 'btn-ghost'}`}
                        style={{ fontSize: '0.7rem', padding: '0.2rem 0.5rem', fontFamily: 'var(--font-mono)' }}
                      >
                        α = {aVal}
                      </button>
                    ))}
                  </div>
                  <div style={{ fontSize: '0.66rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
                    {isAr ? `مضاعف التعلم الفعال (γ): ${(loraAlpha / loraRank).toFixed(1)}x` : `Effective multiplier (γ): ${(loraAlpha / loraRank).toFixed(1)}x`}
                  </div>
                </div>

                {/* Target Attention Modules */}
                <div>
                  <div style={{ fontSize: '0.74rem', fontWeight: 700, color: 'var(--text-secondary)', marginBottom: '0.35rem', display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
                    <span>{isAr ? 'الطبقات المستهدفة بالحقن:' : 'Target Modules:'}</span>
                    <TermTooltip termKey="lora_adapters" size="xs" hideLabel={true} />
                  </div>
                  <div style={{ display: 'flex', gap: '0.35rem', flexWrap: 'wrap' }}>
                    <span className="badge badge-teal" style={{ fontSize: '0.7rem', fontFamily: 'var(--font-mono)' }}>
                      W_q (Query)
                    </span>
                    <span className="badge badge-teal" style={{ fontSize: '0.7rem', fontFamily: 'var(--font-mono)' }}>
                      W_v (Value)
                    </span>
                  </div>
                  <div style={{ fontSize: '0.66rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
                    {isAr ? 'أوزان الاستعلام والقيمة في الانتباه الذاتي' : 'Query & Value projection matrices'}
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Real-time Parameter & Memory Footprint Specs */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '1rem', marginTop: '1rem' }}>
          <div style={{ background: 'var(--bg-secondary)', padding: '1rem', borderRadius: 'var(--radius-sm, 6px)', border: '1px solid var(--border-subtle)' }}>
            <div style={{ color: 'var(--accent-teal)', fontSize: '0.82rem', fontWeight: 800, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span>{isAr ? 'إجمالي المعاملات والأوزان العصبية' : 'Total Trainable Parameters'}</span>
              <TermTooltip termKey="total_parameters" size="sm" hideLabel={true} />
            </div>
            <div style={{ fontSize: '1.5rem', fontWeight: 800, fontFamily: 'var(--font-mono)', color: 'var(--text-primary)', margin: '0.25rem 0' }}>
              {modelStrategy === 'pretrained' ? '~300,000' : (modelEstimate?.total_parameters_formatted || '12,582,914')}
            </div>
            <p style={{ fontSize: '0.72rem', color: 'var(--text-muted)', margin: 0 }}>
              {modelStrategy === 'pretrained' ? (
                <span>
                  {isAr ? 'مصفوفات LoRA النشطة: ~300K | أوزان الأساس المجمدة: ~110M' : 'Active LoRA Adapters: ~300K | Frozen Base: ~110M'}
                </span>
              ) : (
                <span>
                  {isAr ? 'أوزان التضمين' : 'Embeddings'}: {((modelEstimate?.embedding_params || 7813632) / 1e6).toFixed(1)}M | {isAr ? 'طبقات المحول' : 'Encoder'}: {((modelEstimate?.encoder_params || 4768768) / 1e6).toFixed(1)}M
                </span>
              )}
            </p>
          </div>

          <div style={{ background: 'var(--bg-secondary)', padding: '1rem', borderRadius: 'var(--radius-sm, 6px)', border: '1px solid var(--border-subtle)' }}>
            <div style={{ color: 'var(--accent-emerald)', fontSize: '0.82rem', fontWeight: 800, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span>{isAr ? 'استهلاك الذاكرة التقديري (GPU VRAM)' : 'Estimated Memory Footprint'}</span>
              <TermTooltip termKey="memory_footprint" size="sm" hideLabel={true} />
            </div>
            <div style={{ fontSize: '1.5rem', fontWeight: 800, fontFamily: 'var(--font-mono)', color: 'var(--accent-emerald)', margin: '0.25rem 0' }}>
              {modelStrategy === 'pretrained' ? '180 MB' : (modelEstimate?.memory_footprint?.fp16_amp_mb || 24) + ' MB'}
            </div>
            <p style={{ fontSize: '0.72rem', color: 'var(--text-muted)', margin: 0 }}>
              {modelStrategy === 'pretrained' ? (
                <span>{isAr ? 'توفير > 75% من ذاكرة كرت الشاشة مقارنة بالتدريب الكامل' : 'Saves > 75% VRAM compared to full fine-tuning'}</span>
              ) : (
                <span>
                  {isAr ? 'بتسريع FP16 AMP. وفي نمط FP32 الكامل:' : 'FP16 AMP mode. Full FP32:'} ~{modelEstimate?.memory_footprint?.fp32_mb || 48} MB
                </span>
              )}
            </p>
          </div>
        </div>
      </SectionCard>

      {/* Production Python Code Viewer - Apple Developer IDE & Antigravity Syntax Palette */}
      <SectionCard
        sectionCode="03.4"
        title={
          <span style={{ display: 'inline-flex', alignItems: 'center', gap: '0.5rem' }}>
            <span>{isAr ? 'الكود البرمجي المولد آلياً (Executable Python Script)' : 'Generated Python Code'}</span>
            <TermTooltip termKey="reproducibility" size="sm" hideLabel={true} />
          </span>
        }
        subtitle={
          isAr
            ? 'كود بايثون متوافق مع PyTorch ومكتبة dytr، يعكس بدقة نفس الخيارات التي حددتها في الخطوات الثلاث.'
            : 'Standalone PyTorch & dytr script dynamically synthesized to match your 3-step configuration.'
        }
      >
        <div
          style={{
            background: '#0a0d14',
            borderRadius: '12px',
            border: '1px solid rgba(255, 255, 255, 0.12)',
            boxShadow: '0 14px 38px -6px rgba(0, 0, 0, 0.75), 0 0 0 1px rgba(255, 255, 255, 0.05)',
            overflow: 'hidden',
            direction: 'ltr',
            textAlign: 'left',
          }}
        >
          {/* Apple macOS Window Header */}
          <div
            style={{
              background: 'linear-gradient(180deg, #161d2b 0%, #101520 100%)',
              padding: '0.55rem 1rem',
              borderBottom: '1px solid rgba(255, 255, 255, 0.08)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              gap: '1rem',
            }}
          >
            {/* Apple Traffic Light Controls */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '7px' }}>
              <span
                style={{
                  width: '12px',
                  height: '12px',
                  borderRadius: '50%',
                  background: '#FF5F56',
                  border: '1px solid #E0443E',
                  display: 'inline-block',
                  boxShadow: '0 1px 3px rgba(255, 95, 86, 0.35)',
                }}
              />
              <span
                style={{
                  width: '12px',
                  height: '12px',
                  borderRadius: '50%',
                  background: '#FFBD2E',
                  border: '1px solid #DEA123',
                  display: 'inline-block',
                  boxShadow: '0 1px 3px rgba(255, 189, 46, 0.35)',
                }}
              />
              <span
                style={{
                  width: '12px',
                  height: '12px',
                  borderRadius: '50%',
                  background: '#27C93F',
                  border: '1px solid #1AAB29',
                  display: 'inline-block',
                  boxShadow: '0 1px 3px rgba(39, 201, 63, 0.35)',
                }}
              />
            </div>

            {/* macOS Tab / Active Document Badge */}
            <div
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '0.45rem',
                background: 'rgba(255, 255, 255, 0.05)',
                border: '1px solid rgba(255, 255, 255, 0.1)',
                padding: '0.22rem 0.85rem',
                borderRadius: '6px',
                fontFamily: '"SF Mono", "JetBrains Mono", "Fira Code", monospace',
                fontSize: '0.74rem',
                color: '#e2e8f0',
                letterSpacing: '0.02em',
              }}
            >
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                <polyline points="14 2 14 8 20 8"></polyline>
                <line x1="16" y1="13" x2="8" y2="13"></line>
                <line x1="16" y1="17" x2="8" y2="17"></line>
                <polyline points="10 9 9 9 8 9"></polyline>
              </svg>
              <span style={{ fontWeight: 600 }}>model_architecture.py</span>
              <span style={{ width: '6px', height: '6px', borderRadius: '50%', background: '#38bdf8', opacity: 0.8 }} />
            </div>

            {/* Apple-style Copy Button */}
            <button
              type="button"
              onClick={handleCopyCode}
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '0.4rem',
                fontSize: '0.72rem',
                fontWeight: 600,
                padding: '0.28rem 0.75rem',
                borderRadius: '6px',
                background: copySuccess ? 'rgba(34, 197, 94, 0.15)' : 'rgba(255, 255, 255, 0.08)',
                color: copySuccess ? '#4ade80' : '#cbd5e1',
                border: copySuccess ? '1px solid rgba(74, 222, 128, 0.35)' : '1px solid rgba(255, 255, 255, 0.15)',
                cursor: 'pointer',
                transition: 'all 0.18s ease-in-out',
              }}
            >
              {copySuccess ? (
                <>
                  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                    <polyline points="20 6 9 17 4 12"></polyline>
                  </svg>
                  <span>{isAr ? 'تم النسخ!' : 'Copied!'}</span>
                </>
              ) : (
                <>
                  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                    <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                  </svg>
                  <span>{isAr ? 'نسخ الكود' : 'Copy Code'}</span>
                </>
              )}
            </button>
          </div>

          {/* Code Editor Body with Line Numbers & Antigravity Syntax Palette */}
          <div
            style={{
              padding: '0.85rem 0',
              maxHeight: '300px',
              overflowX: 'auto',
              overflowY: 'auto',
              fontFamily: '"SF Mono", "JetBrains Mono", "Fira Code", Menlo, Consolas, monospace',
              fontSize: '0.78rem',
              lineHeight: 1.6,
            }}
          >
            {(livePythonCode || '').split('\n').map((line, lineIdx) => {
              const hashIndex = line.indexOf('#');
              const codePart = hashIndex !== -1 ? line.slice(0, hashIndex) : line;
              const commentPart = hashIndex !== -1 ? line.slice(hashIndex) : '';

              const tokenRegex = /("(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|\b(?:from|import|def|class|return|if|else|elif|for|in|as|with|pass|lambda|sum|print)\b|\b(?:True|False|None)\b|\b(?:DynamicTransformer|ModelConfig|TaskConfig|TrainingStrategy|PretrainedBackbone|LoRAConfig|TaskHead|SENTENCE_CLASSIFICATION|CAUSAL_LM|TOKEN_CLASSIFICATION|LoRAAdapter|Embedding|Linear|AdamW)\b|\b\d+(?:\.\d+)?\b|[a-zA-Z_]\w*(?=\()|[a-zA-Z_]\w*(?==)|[a-zA-Z_]\w*|[^\s\w])/g;

              const tokens = [];
              let match;
              let lastIndex = 0;

              while ((match = tokenRegex.exec(codePart)) !== null) {
                if (match.index > lastIndex) {
                  tokens.push({ type: 'text', value: codePart.slice(lastIndex, match.index) });
                }
                const val = match[0];
                let type = 'text';

                if (val.startsWith('"') || val.startsWith("'")) {
                  type = 'string';
                } else if (/^(from|import|def|class|return|if|else|elif|for|in|as|with|pass|lambda|sum|print)$/.test(val)) {
                  type = 'keyword';
                } else if (/^(True|False|None)$/.test(val)) {
                  type = 'boolean';
                } else if (/^(DynamicTransformer|ModelConfig|TaskConfig|TrainingStrategy|PretrainedBackbone|LoRAConfig|TaskHead|SENTENCE_CLASSIFICATION|CAUSAL_LM|TOKEN_CLASSIFICATION|LoRAAdapter|Embedding|Linear|AdamW)$/.test(val)) {
                  type = 'class';
                } else if (/^\d+(\.\d+)?$/.test(val)) {
                  type = 'number';
                } else if (codePart[match.index + val.length] === '(') {
                  type = 'function';
                } else if (codePart[match.index + val.length] === '=') {
                  type = 'parameter';
                }

                tokens.push({ type, value: val });
                lastIndex = tokenRegex.lastIndex;
              }

              if (lastIndex < codePart.length) {
                tokens.push({ type: 'text', value: codePart.slice(lastIndex) });
              }

              const getTokenColor = (tokenType) => {
                switch (tokenType) {
                  case 'keyword': return '#38bdf8'; // Antigravity electric cyan
                  case 'class': return '#f43f5e'; // Antigravity vivid coral/rose
                  case 'function': return '#c084fc'; // Antigravity purple/violet
                  case 'parameter': return '#7dd3fc'; // Soft sky blue
                  case 'string': return '#4ade80'; // Emerald lime
                  case 'number':
                  case 'boolean': return '#fbbf24'; // Amber gold
                  default: return '#e2e8f0'; // Crisp off-white
                }
              };

              return (
                <div key={lineIdx} style={{ display: 'flex', minHeight: '1.45rem', alignItems: 'center' }}>
                  {/* Line Number Gutter */}
                  <span
                    style={{
                      width: '3.2rem',
                      flexShrink: 0,
                      textAlign: 'right',
                      paddingRight: '1rem',
                      color: '#475569',
                      userSelect: 'none',
                      fontSize: '0.72rem',
                      borderRight: '1px solid rgba(255, 255, 255, 0.06)',
                    }}
                  >
                    {lineIdx + 1}
                  </span>
                  {/* Code Line */}
                  <span style={{ flex: 1, paddingLeft: '1rem', whiteSpace: 'pre', overflowX: 'visible' }}>
                    {tokens.map((tok, tokIdx) => (
                      <span
                        key={tokIdx}
                        style={{
                          color: getTokenColor(tok.type),
                          fontWeight: tok.type === 'keyword' || tok.type === 'class' ? 600 : 400,
                        }}
                      >
                        {tok.value}
                      </span>
                    ))}
                    {commentPart && (
                      <span style={{ color: '#64748b', fontStyle: 'italic' }}>
                        {commentPart}
                      </span>
                    )}
                  </span>
                </div>
              );
            })}
          </div>

          {/* Apple IDE Footer / Status Bar */}
          <div
            style={{
              background: '#0d121c',
              borderTop: '1px solid rgba(255, 255, 255, 0.06)',
              padding: '0.35rem 1rem',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              fontSize: '0.68rem',
              color: '#64748b',
              fontFamily: '"SF Mono", "JetBrains Mono", monospace',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem' }}>
              <span style={{ color: '#38bdf8', fontWeight: 600 }}>Python 3.10+</span>
              <span>•</span>
              <span>dytr v1.0</span>
              <span>•</span>
              <span>PyTorch 2.4</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem' }}>
              <span>UTF-8</span>
              <span>•</span>
              <span>Spaces: 4</span>
              <span>•</span>
              <span style={{ color: '#4ade80' }}>● Executable</span>
            </div>
          </div>
        </div>
      </SectionCard>

      {/* Stage Transition & Pipeline Handoff Visual Bridge - Theme Adaptive */}
      <div
        style={{
          marginTop: '2.5rem',
          background: 'var(--bg-elevated)',
          border: '1px solid var(--border-medium)',
          borderRadius: 'var(--radius-lg, 14px)',
          boxShadow: 'var(--shadow-md)',
          padding: '1.5rem',
          position: 'relative',
        }}
      >
        {/* Top Header with Progress Badge */}
        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            flexWrap: 'wrap',
            gap: '0.75rem',
            marginBottom: '1.25rem',
            borderBottom: '1px solid var(--border-subtle)',
            paddingBottom: '0.85rem',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
            <div
              style={{
                width: '28px',
                height: '28px',
                borderRadius: '8px',
                background: 'var(--accent-teal-glow)',
                border: '1px solid var(--accent-teal)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: 'var(--accent-teal)',
                fontWeight: 800,
                fontSize: '0.85rem',
              }}
            >
              <Check size={14} />
            </div>
            <div>
              <h3 style={{ margin: 0, fontSize: '0.95rem', fontWeight: 800, color: 'var(--text-primary)' }}>
                {isAr ? 'عقد اعتماد النموذج والتوجيه للمرحلة التالية' : 'Model Approval & Pipeline Handoff'}
              </h3>
              <p style={{ margin: 0, fontSize: '0.72rem', color: 'var(--text-muted)' }}>
                {isAr ? 'ربط مخرجات الخطوات السابقة بالوجهة القادمة بوضوح وبساطة' : 'Connecting past outputs to the upcoming destination clearly'}
              </p>
            </div>
          </div>
          <span className="badge badge-teal" style={{ fontSize: '0.7rem', padding: '0.3rem 0.75rem', fontWeight: 700 }}>
            {isAr ? 'المرحلة 03 مكتملة بنجاح' : 'Stage 03 Complete'}
          </span>
        </div>

        {/* 3-Step Visual Connection Strip (السابق -> الحالي -> التالي) */}
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
            gap: '0.85rem',
            marginBottom: '1.4rem',
          }}
        >
          {/* Card 1: السابق (Stage 02) */}
          <div
            style={{
              background: 'var(--bg-secondary)',
              border: '1px solid var(--border-subtle)',
              borderRadius: 'var(--radius-md, 10px)',
              padding: '0.85rem 1rem',
              boxShadow: 'var(--shadow-sm)',
            }}
          >
            <div
              style={{
                fontSize: '0.68rem',
                color: 'var(--text-secondary)',
                fontWeight: 700,
                textTransform: 'uppercase',
                marginBottom: '0.3rem',
                display: 'flex',
                alignItems: 'center',
                gap: '0.35rem',
              }}
            >
              <span style={{ color: 'var(--accent-emerald)' }}>●</span>
              <span>{isAr ? 'الخطوة السابقة: تجهيز البيانات (02)' : 'Past: Data Prep (02)'}</span>
            </div>
            <div
              style={{
                fontSize: '0.84rem',
                fontWeight: 800,
                color: 'var(--text-primary)',
                whiteSpace: 'nowrap',
                overflow: 'hidden',
                textOverflow: 'ellipsis',
              }}
            >
              {datasetName || (isAr ? 'بيانات جاهزة ومطابقة' : 'Dataset Ready')}
            </div>
            <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
              {isAr ? 'تم فحص الأعمدة وتنظيف النصوص المستهدفة' : 'Target columns verified & texts cleaned'}
            </div>
          </div>

          {/* Card 2: الحالي (Stage 03) */}
          <div
            style={{
              background: 'var(--bg-secondary)',
              border: '1.5px solid var(--accent-teal)',
              borderRadius: 'var(--radius-md, 10px)',
              padding: '0.85rem 1rem',
              boxShadow: 'var(--shadow-sm)',
            }}
          >
            <div
              style={{
                fontSize: '0.68rem',
                color: 'var(--accent-teal)',
                fontWeight: 700,
                textTransform: 'uppercase',
                marginBottom: '0.3rem',
                display: 'flex',
                alignItems: 'center',
                gap: '0.35rem',
              }}
            >
              <span style={{ color: 'var(--accent-teal)' }}>●</span>
              <span>{isAr ? 'الخطوة الحالية: معمارية النموذج (03)' : 'Current: Architecture (03)'}</span>
            </div>
            <div style={{ fontSize: '0.84rem', fontWeight: 800, color: 'var(--text-primary)' }}>
              {modelEstimate?.total_parameters_formatted || '12.5M'} {isAr ? 'معلمة مضبوطة' : 'configured params'}
            </div>
            <div style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', marginTop: '0.25rem' }}>
              {modelStrategy === 'pretrained' ? `تكييف LoRA (${selectedPretrainedModel.toUpperCase()})` : (isAr ? 'بناء من الصفر (Native DyTr)' : 'From Scratch')} • {selectedTaskType === 'sentence_classification' ? (isAr ? 'تصنيف جمل' : 'Classification') : selectedTaskType}
            </div>
          </div>

          {/* Card 3: التالي (Stage 04) */}
          <div
            style={{
              background: 'var(--bg-secondary)',
              border: '1.5px solid var(--accent-cyan)',
              borderRadius: 'var(--radius-md, 10px)',
              padding: '0.85rem 1rem',
              boxShadow: 'var(--shadow-sm)',
            }}
          >
            <div
              style={{
                fontSize: '0.68rem',
                color: 'var(--accent-cyan)',
                fontWeight: 700,
                textTransform: 'uppercase',
                marginBottom: '0.3rem',
                display: 'flex',
                alignItems: 'center',
                gap: '0.35rem',
              }}
            >
              <ArrowRight size={14} style={{ color: 'var(--accent-cyan)', transform: isAr ? 'scaleX(-1)' : 'none' }} />
              <span>{isAr ? 'الخطوة القادمة: التدريب (04)' : 'Next: Training Arena (04)'}</span>
            </div>
            <div style={{ fontSize: '0.84rem', fontWeight: 800, color: 'var(--text-primary)' }}>
              {isAr ? 'بدء التعلم ومراقبة دقة النموذج لحظياً' : 'Launch Training & Live Telemetry'}
            </div>
            <div style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', marginTop: '0.25rem' }}>
              {isAr ? 'تحديد سرعة التعلم والضغط على «ابدأ التدريب»' : 'Set learning rate & press Launch'}
            </div>
          </div>
        </div>

        {/* Actions Bottom Bar */}
        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            flexWrap: 'wrap',
            gap: '1rem',
            borderTop: '1px solid var(--border-subtle)',
            paddingTop: '1rem',
          }}
        >
          <button
            type="button"
            onClick={() => setActiveModule('dataprep')}
            className="btn btn-secondary"
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: '0.5rem',
              padding: '0.6rem 1.15rem',
              fontSize: '0.8rem',
              fontWeight: 700,
              cursor: 'pointer',
            }}
          >
            <ArrowLeft size={16} style={{ transform: isAr ? 'scaleX(-1)' : 'none' }} />
            <span>{isAr ? 'مراجعة تجهيز البيانات (المرحلة 02)' : 'Review Data Prep (Stage 02)'}</span>
          </button>

          <button
            type="button"
            onClick={() => {
              const textToCommit = selectedTextCol || datasetAnalysis.defaultTextCol || 'text';
              const labelToCommit = selectedLabelCol || datasetAnalysis.defaultLabelCol || 'label';
              const updatedTasks = (modelConfig.tasks || []).map((t, idx) =>
                idx === 0 ? { ...t, text_column: textToCommit, label_column: labelToCommit } : t
              );
              if (updatedTasks.length === 0) {
                updatedTasks.push({
                  task_name: 'task_classification',
                  task_type: 'sentence_classification',
                  strategy: 'SENTENCE_CLASSIFICATION',
                  num_classes: targetNumClasses,
                  text_column: textToCommit,
                  label_column: labelToCommit,
                });
              }
              updateModelConfig({ tasks: updatedTasks });
              setActiveModule('arena');
            }}
            className="btn btn-primary"
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: '0.6rem',
              padding: '0.7rem 1.6rem',
              fontSize: '0.85rem',
              fontWeight: 800,
              cursor: 'pointer',
            }}
          >
            <span>{isAr ? 'الانتقال إلى التدريب والبدء (المرحلة 04)' : 'Proceed to Training Arena & Launch (Stage 04)'}</span>
            <ArrowRight size={16} style={{ transform: isAr ? 'scaleX(-1)' : 'none' }} />
          </button>
        </div>
      </div>
    </div>
  );
}
