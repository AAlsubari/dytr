// ==============================================================================
// DyTrStudio Data Mining - Phase 1: Comprehensive Data Acquisition Studio
// 5 Modalities: File Ingest, Web Scraping, Scientific Search, Synthetic Engine & Benchmarks
// Zero Emojis Policy - High Rigor Arabic & English Educational Guidance - Dual Theme Ready
// ==============================================================================

import React, { useState, useRef, useEffect, useMemo } from 'react';
import { useDatasetStore } from '../../store/useDatasetStore';
import { useStudioStore } from '../../store/studioStore';
import { api } from '../../utils/api';
import {
  sampleDatasetsCatalog,
  generateArabicSentiment,
  generateArabicNews,
  generateIrisData,
  generateTitanicData,
  generateDomainAcademicSample,
} from '../../lib/data/synthetic';
import TermTooltip from '../common/TermTooltip';
import {
  X,
  ExternalLink,
  ArrowRight,
  Zap,
  Check,
  ChevronLeft,
  ChevronRight,
  AlertCircle,
  Database,
  Sparkles,
  Layers,
  Search,
  RefreshCw,
  Globe,
  Play,
  CheckCircle2,
  Sliders,
  ListFilter,
  CheckCircle,
  Trash2,
  AlertTriangle,
  Copy,
  RotateCcw,
  Bookmark,
  ChevronUp,
  ChevronDown,
} from 'lucide-react';
import { showStudioAlert } from '../common/PlatformAlertModal';

export function parseRawTabularData(text, url = '') {
  if (!text || !text.trim()) return null;

  const trimmed = text.trim();

  // 1. JSON parsing support (handles JSON arrays or { rows: [...] } / { data: [...] })
  if (trimmed.startsWith('{') || trimmed.startsWith('[')) {
    try {
      const parsedJson = JSON.parse(trimmed);
      let rowsArray = [];
      if (Array.isArray(parsedJson)) {
        rowsArray = parsedJson;
      } else if (parsedJson.rows && Array.isArray(parsedJson.rows)) {
        rowsArray = parsedJson.rows.map((r) => r.row || r);
      } else if (parsedJson.data && Array.isArray(parsedJson.data)) {
        rowsArray = parsedJson.data;
      }
      if (rowsArray.length > 0 && typeof rowsArray[0] === 'object' && rowsArray[0] !== null) {
        const headers = Object.keys(rowsArray[0]);
        const cleanRows = rowsArray.map((r) => {
          const cleanObj = {};
          headers.forEach((h) => {
            let v = r[h];
            if (v && typeof v === 'object') {
              v = v.src || v.url || v.path || JSON.stringify(v);
            }
            cleanObj[h] = v !== undefined ? v : '';
          });
          return cleanObj;
        });
        const rawName = url.split('/').pop()?.split('?')[0] || 'dataset.json';
        return {
          status: 'success',
          filename: rawName.endsWith('.csv') || rawName.endsWith('.json') ? rawName : `${rawName}.csv`,
          rows_count: cleanRows.length,
          cols_count: headers.length,
          columns: headers,
          rows: cleanRows,
        };
      }
    } catch (e) {
      // not JSON, fallback to tabular parser
    }
  }

  const lines = text.split(/\r?\n/).filter((l) => l.trim().length > 0);
  if (lines.length === 0) return null;

  // 2. Markdown Table parsing support (| col1 | col2 |)
  if (trimmed.includes('|') && lines.some((l) => l.includes('---'))) {
    const tableLines = lines.filter((l) => l.includes('|'));
    const separatorIdx = tableLines.findIndex(
      (l) => l.replace(/[^|\-:]/g, '').length >= 3 && l.includes('---')
    );
    if (separatorIdx > 0) {
      const headerLine = tableLines[separatorIdx - 1];
      const headers = headerLine
        .split('|')
        .map((c) => c.trim())
        .filter((c, i, a) => (i > 0 && i < a.length - 1) || c.length > 0);
      const dataRows = [];
      for (let i = separatorIdx + 1; i < tableLines.length; i++) {
        const cells = tableLines[i]
          .split('|')
          .map((c) => c.trim())
          .filter((c, idx, a) => (idx > 0 && idx < a.length - 1) || c.length > 0);
        if (cells.length === headers.length) {
          const rowObj = {};
          headers.forEach((h, ci) => {
            rowObj[h] = cells[ci] || '';
          });
          dataRows.push(rowObj);
        }
      }
      if (dataRows.length > 0) {
        return {
          status: 'success',
          filename: 'curated_markdown_table.csv',
          rows_count: dataRows.length,
          cols_count: headers.length,
          columns: headers,
          rows: dataRows,
        };
      }
    }
  }

  // 3. Delimited CSV / TSV / Semicolon parsing
  const firstLine = lines[0];
  const isTsv = url.endsWith('.tsv') || (firstLine.includes('\t') && !firstLine.includes(','));
  const delimiter = isTsv ? '\t' : (firstLine.includes(';') && !firstLine.includes(',') ? ';' : ',');

  const rawFirstCols = firstLine.split(delimiter).map((c) => c.trim().replace(/^"|"$/g, ''));
  const firstValLower = (rawFirstCols[0] || '').toLowerCase();

  const isDataRow = rawFirstCols.length >= 2 && (
    ['ham', 'spam', 'pos', 'neg', 'positive', 'negative', 'true', 'false'].includes(firstValLower) ||
    (rawFirstCols[1] || '').length > 35 ||
    (rawFirstCols[0] || '').length > 35 ||
    (!isNaN(Number(rawFirstCols[0])) && rawFirstCols[0] !== '')
  );

  let headers = [];
  let dataLines = [];

  if (isDataRow) {
    if (rawFirstCols.length === 2 && ['ham', 'spam', 'pos', 'neg', 'positive', 'negative', '0', '1'].includes(firstValLower)) {
      headers = ['label', 'text'];
    } else if (rawFirstCols.length === 3 && ['ham', 'spam', 'pos', 'neg'].includes(firstValLower)) {
      headers = ['label', 'text', 'id'];
    } else {
      headers = rawFirstCols.map((_, i) => `col_${i + 1}`);
    }
    dataLines = lines;
  } else {
    headers = rawFirstCols.map((h, i) => h || `col_${i + 1}`);
    dataLines = lines.slice(1);
  }

  const rows = dataLines.map((line) => {
    const values = line.split(delimiter).map((v) => v.trim().replace(/^"|"$/g, ''));
    const obj = {};
    headers.forEach((h, i) => {
      let val = values[i] !== undefined ? values[i] : '';
      if (val !== '' && !isNaN(Number(val))) {
        val = Number(val);
      }
      obj[h] = val;
    });
    return obj;
  });

  const rawName = url.split('/').pop()?.split('?')[0] || (isTsv ? 'dataset.tsv' : 'dataset.csv');
  const filename = rawName.endsWith('.csv') || rawName.endsWith('.tsv') ? rawName : `${rawName}.csv`;

  return {
    status: 'success',
    filename,
    rows_count: rows.length,
    cols_count: headers.length,
    columns: headers,
    rows: rows,
  };
}

export default function AcquisitionPanel() {
  const { language } = useStudioStore();
  const isAr = language === 'ar';
  const { setDataset, savedLibrary, deleteFromLibrary } = useDatasetStore();

  const [activeSource, setActiveSource] = useState('upload'); // 'upload' | 'scrape' | 'search' | 'synthetic' | 'samples' | 'library'
  const [showHelper, setShowHelper] = useState(false);
  const [librarySearch, setLibrarySearch] = useState('');

  const filteredLibrary = useMemo(() => {
    if (!savedLibrary || savedLibrary.length === 0) return [];
    if (!librarySearch.trim()) return savedLibrary;
    const q = librarySearch.toLowerCase().trim();
    return savedLibrary.filter((item) => {
      const nameMatch = item.name?.toLowerCase().includes(q);
      const descMatch = item.description?.toLowerCase().includes(q);
      const tagMatch = item.tags?.some((t) => t.toLowerCase().includes(q));
      return nameMatch || descMatch || tagMatch;
    });
  }, [savedLibrary, librarySearch]);

  // --- 1. File Upload State ---
  const [uploadError, setUploadError] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [uploadedPreview, setUploadedPreview] = useState(null);

  // Platform Native Alert Modal Trigger (Unified Studio-wide Alert System)
  const showPlatformAlert = (title, message, type = 'error') => {
    showStudioAlert(title, message, type);
  };

  // Utility to clean pasted and concatenated URLs
  const cleanPastedUrl = (raw) => {
    if (!raw) return '';
    let str = String(raw).trim();
    const matches = [...str.matchAll(/https?:\/\//gi)];
    if (matches.length > 1) {
      str = str.slice(matches[matches.length - 1].index);
    } else if (matches.length === 1 && matches[0].index > 0) {
      str = str.slice(matches[0].index);
    }
    if (str.includes('?')) {
      const [base, query] = str.split('?', 2);
      if (query && query.includes('?')) {
        const cleanQ = query.split('?')[0].replace(/(\d+)[a-zA-Z_]+$/, '$1');
        str = `${base}?${cleanQ}`;
      }
    }
    return str.trim();
  };

  const getDomainFromUrl = (url) => {
    try {
      if (!url) return '';
      const u = new URL(url.startsWith('http') ? url : 'https://' + url);
      return u.hostname.replace(/^www\./, '');
    } catch {
      return '';
    }
  };

  // --- 2. Web Scraping State ---
  const [scrapeType, setScrapeType] = useState('multipage'); // 'single' | 'multipage'
  const [scrapeUrl, setScrapeUrl] = useState('https://arxiv.org/list/cs.AI/recent');
  const [scrapeMode, setScrapeMode] = useState('auto'); // 'auto' | 'tables' | 'text'
  const [scrapeLoading, setScrapeLoading] = useState(false);
  const [scrapedPreview, setScrapedPreview] = useState(null);
  const [scrapedPreviewOpen, setScrapedPreviewOpen] = useState(true);
  const [scrapedSearch, setScrapedSearch] = useState('');
  const [scrapedPage, setScrapedPage] = useState(1);

  // Multi-Page Pattern Scraper State (Initialized empty with clean placeholders)
  const [patternMethod, setPatternMethod] = useState('diff'); // 'diff' | 'auto' | 'template'
  const [diffUrl1, setDiffUrl1] = useState('');
  const [diffUrl2, setDiffUrl2] = useState('');
  const [autoStartUrl, setAutoStartUrl] = useState('');
  const [manualTemplate, setManualTemplate] = useState('');
  const [inferLoading, setInferLoading] = useState(false);
  const [inferredPattern, setInferredPattern] = useState(null);
  const [inferError, setInferError] = useState(null);
  const [crawlStartPage, setCrawlStartPage] = useState(1);
  const [crawlNumPages, setCrawlNumPages] = useState(5);
  const [crawlStep, setCrawlStep] = useState(1);
  const [crawlExtractMode, setCrawlExtractMode] = useState('auto'); // 'auto' | 'tables' | 'cards' | 'text'
  const [crawlDelay, setCrawlDelay] = useState(1.5); // Polite Delay (Seconds)
  const [crawlWaitTime, setCrawlWaitTime] = useState(2.0); // Dynamic JS Render Wait (Seconds)
  const [crawlUseStealth, setCrawlUseStealth] = useState(true); // Stealth Headless Driver
  const [crawlEnableJitter, setCrawlEnableJitter] = useState(true); // Anti-Ban Random Jitter
  const [singleWaitTime, setSingleWaitTime] = useState(2.0);
  const [singleUseStealth, setSingleUseStealth] = useState(true);
  const [singleScrapeError, setSingleScrapeError] = useState(null);
  const [scrapeElapsedSeconds, setScrapeElapsedSeconds] = useState(0);
  const [scrapeTotalDuration, setScrapeTotalDuration] = useState(null);
  const scrapeTimerRef = useRef(null);
  const [multiScrapeLoading, setMultiScrapeLoading] = useState(false);
  const [multiScrapeStats, setMultiScrapeStats] = useState(null);
  const [multiScrapeError, setMultiScrapeError] = useState(null);

  useEffect(() => {
    return () => {
      if (scrapeTimerRef.current) clearInterval(scrapeTimerRef.current);
    };
  }, []);

  // Curated Single-Page Presets (Instant 1-Click Verification)
  const singlePagePresets = [
    {
      id: 'arxiv_single',
      name_ar: 'أحدث أبحاث الذكاء الاصطناعي (arXiv Recent)',
      name_en: 'arXiv AI Recent Page',
      url: 'https://arxiv.org/list/cs.AI/recent',
      badge: 'arXiv AI',
    },
    {
      id: 'telegram_single',
      name_ar: 'قناة تيليجرام العامة (Telegram Channel Feed)',
      name_en: 'Telegram Channel Feed',
      url: 'https://t.me/s/Jamal_Almansoub_Channel',
      badge: 'Telegram',
    },
    {
      id: 'wikipedia_single',
      name_ar: 'فهرس مقالات التعلم الآلي (Wikipedia ML)',
      name_en: 'Wikipedia ML Articles',
      url: 'https://en.wikipedia.org/w/index.php?title=Special:WhatLinksHere/Machine_learning&limit=50',
      badge: 'Wikipedia',
    },
    {
      id: 'ncbi_single',
      name_ar: 'ملخصات الأبحاث الطبية (NCBI PubMed/PMC)',
      name_en: 'NCBI PMC Research',
      url: 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pmc&term=cancer&retmax=20',
      badge: 'NCBI / PMC',
    },
  ];

  // Curated Multi-Page Presets (Scientific, Academic, and Technical Public Datasets)
  const multiPagePresets = [
    {
      id: 'arxiv_ai',
      name_ar: 'أبحاث الذكاء الاصطناعي (arXiv AI Recent Papers)',
      name_en: 'arXiv AI Papers Catalog',
      url1: 'https://arxiv.org/list/cs.AI/recent?skip=0&show=25',
      url2: 'https://arxiv.org/list/cs.AI/recent?skip=25&show=25',
      template: 'https://arxiv.org/list/cs.AI/recent?skip={page}&show=25',
      start: 0,
      step: 25,
      numPages: 4,
      extractMode: 'auto',
      badge: 'arXiv Papers',
    },
    {
      id: 'telegram_channel',
      name_ar: 'قناة تيليجرام العامة (Telegram Public Feed)',
      name_en: 'Telegram Channel Public Feed',
      url1: 'https://t.me/s/Jamal_Almansoub_Channel?before=100',
      url2: 'https://t.me/s/Jamal_Almansoub_Channel?before=80',
      template: 'https://t.me/s/Jamal_Almansoub_Channel?before={page}',
      start: 100,
      step: 20,
      numPages: 3,
      extractMode: 'auto',
      badge: 'Telegram Channel',
    },
    {
      id: 'wikipedia_ml',
      name_ar: 'فهرس مقالات ويكيبيديا (Wikipedia Machine Learning)',
      name_en: 'Wikipedia Articles Index',
      url1: 'https://en.wikipedia.org/w/index.php?title=Special:WhatLinksHere/Machine_learning&offset=0&limit=50',
      url2: 'https://en.wikipedia.org/w/index.php?title=Special:WhatLinksHere/Machine_learning&offset=50&limit=50',
      template: 'https://en.wikipedia.org/w/index.php?title=Special:WhatLinksHere/Machine_learning&offset={page}&limit=50',
      start: 0,
      step: 50,
      numPages: 3,
      extractMode: 'auto',
      badge: 'Wikipedia Index',
    },
    {
      id: 'ncbi_pmc',
      name_ar: 'الأبحاث الطبية والبيولوجية (NCBI PubMed/PMC)',
      name_en: 'NCBI Biomedical Research Archive',
      url1: 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pmc&term=cancer&retstart=0&retmax=20',
      url2: 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pmc&term=cancer&retstart=20&retmax=20',
      template: 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pmc&term=cancer&retstart={page}&retmax=20',
      start: 0,
      step: 20,
      numPages: 3,
      extractMode: 'auto',
      badge: 'Biomedical / PMC',
    },
  ];

  const applyMultiPagePreset = (preset) => {
    setDiffUrl1(preset.url1);
    setDiffUrl2(preset.url2);
    setAutoStartUrl(preset.url1);
    setManualTemplate(preset.template);
    setCrawlStartPage(preset.start);
    setCrawlStep(preset.step);
    setCrawlNumPages(preset.numPages);
    setCrawlExtractMode(preset.extractMode);
    setInferError(null);

    const sample_urls = [
      preset.template.replace('{page}', String(preset.start)),
      preset.template.replace('{page}', String(preset.start + preset.step)),
      preset.template.replace('{page}', String(preset.start + 2 * preset.step)),
    ];

    setInferredPattern({
      status: 'success',
      detected_param: preset.step > 1 ? 'offset_param' : 'page_param',
      param_type: 'verified_preset',
      start_val: preset.start,
      step: preset.step,
      template: preset.template,
      sample_urls: sample_urls,
      method: 'academic_preset',
      explanation_ar: `تم اعتماد نموذج (${isAr ? preset.name_ar : preset.name_en}) بنجاح. يبدأ من ${preset.start} بخطوة ${preset.step}.`,
      explanation_en: `Preset verified (${preset.name_en}): starts at ${preset.start} with step ${preset.step}.`,
    });
  };

  // --- 3. Scientific Web Search State ---
  const [searchQuery, setSearchQuery] = useState('SMS Spam Collection');
  const [searchLoading, setSearchLoading] = useState(false);
  const [searchResults, setSearchResults] = useState([]);
  const [searchPlatformFilters, setSearchPlatformFilters] = useState(['all']); // Multi-select array: ['all'] or ['uci', 'datagov', ...]
  const [searchPage, setSearchPage] = useState(1);
  const [searchLimit, setSearchLimit] = useState(20); // User-configurable dataset count (userinput_n)
  const [searchError, setSearchError] = useState(null);
  const SEARCH_PAGE_SIZE = 4;
  const [activePlatformInfo, setActivePlatformInfo] = useState(null); // Platform guide modal dialog
  const [previewingDataset, setPreviewingDataset] = useState(null);
  const [fetchingRemote, setFetchingRemote] = useState(false);
  const [remotePreviewResult, setRemotePreviewResult] = useState(null);
  const [searchSuccessMsg, setSearchSuccessMsg] = useState(null);

  const handleTogglePlatformFilter = (id) => {
    setSearchPage(1);
    if (id === 'all') {
      setSearchPlatformFilters(['all']);
      return;
    }
    let current = searchPlatformFilters.filter((p) => p !== 'all');
    if (current.includes(id)) {
      current = current.filter((p) => p !== id);
    } else {
      current.push(id);
    }
    if (current.length === 0) {
      current = ['all'];
    }
    setSearchPlatformFilters(current);
  };

  React.useEffect(() => {
    if (activeSource === 'search' && searchResults.length === 0 && !searchLoading) {
      handleSearch();
    }
  }, [activeSource]);

  // --- 4. Synthetic Generator State ---
  const [syntheticTab, setSyntheticTab] = useState('presets'); // 'presets' | 'guided' | 'manual'
  const [syntheticType, setSyntheticType] = useState('sentiment');
  const [syntheticCount, setSyntheticCount] = useState(60);

  // Guided Builder State
  const [customDatasetName, setCustomDatasetName] = useState('بيانات_اصطناعية_مخصصة.csv');
  const [customRowsCount, setCustomRowsCount] = useState(50);
  const [customColumns, setCustomColumns] = useState([
    { id: 1, name: 'review_text', type: 'arabic_text' },
    { id: 2, name: 'sentiment_label', type: 'category', categories: 'إيجابي, سلبي, محايد' },
    { id: 3, name: 'confidence_score', type: 'numeric', min: 0.5, max: 1.0 },
    { id: 4, name: 'record_date', type: 'date' },
  ]);

  // Manual Matrix Entry State
  const [manualText, setManualText] = useState(
`text,sentiment,rating,category
"معمارية المحول رائعة وأظهرت استقراراً مبهراً في التدريب.",Positive,5,NLP
"دقة النموذج تجاوزت التوقعات مع انخفاض ملحوظ في دالة الخسارة.",Positive,5,Optimization
"حدث تذبذب شديد في التدرجات وانفجار في الأوزان أثناء التعلم.",Negative,1,Gradients
"انخفضت الدقة على المهمة القديمة بسبب ظاهرة النسيان الكارثي.",Negative,2,Continual
"تم تحديث مصفوفة الأوزان وفق معامل الانحدار الافتراضي.",Neutral,3,General`
  );

  // =========================================================================
  // HANDLERS
  // =========================================================================

  // 1. File Upload Processor
  const processUploadedFile = (file) => {
    if (!file) return;
    setUploadError(null);
    const reader = new FileReader();
    const isJson = file.name.endsWith('.json');

    reader.onload = (evt) => {
      try {
        const text = evt.target?.result;
        if (isJson) {
          const parsed = JSON.parse(text);
          const arrayData = Array.isArray(parsed) ? parsed : [parsed];
          setUploadedPreview({
            data: arrayData,
            name: file.name,
            size: file.size,
            rowsCount: arrayData.length,
            colsCount: Object.keys(arrayData[0] || {}).length,
          });
        } else {
          const parsed = parseRawTabularData(text, file.name);
          if (parsed && parsed.rows && parsed.rows.length > 0) {
            setUploadedPreview({
              data: parsed.rows,
              name: file.name,
              size: file.size,
              rowsCount: parsed.rows.length,
              colsCount: parsed.cols_count,
            });
          } else {
            setUploadError(isAr ? 'الملف فارغ أو لا يحتوي على صفوف بيانات.' : 'File is empty or missing data rows.');
          }
        }
      } catch (err) {
        setUploadError(isAr ? 'تعذر تحليل الملف. يرجى التحقق من صحة الصيغة.' : 'Failed to parse file.');
      }
    };

    reader.readAsText(file);
  };

  const handleFileUpload = (e) => {
    const file = e.target.files?.[0];
    if (file) processUploadedFile(file);
  };

  const handleConfirmUpload = () => {
    if (uploadedPreview && uploadedPreview.data) {
      setDataset(uploadedPreview.data, uploadedPreview.name, 'upload', { size: uploadedPreview.size });
      setUploadedPreview(null);
    }
  };

  // 2. Real Web Scraping Handler
  const handleScrape = async () => {
    const raw = (scrapeUrl || '').trim();
    if (!raw) return;
    const cleanUrl = cleanPastedUrl(raw);
    setScrapeLoading(true);
    setSingleScrapeError(null);
    setScrapeElapsedSeconds(0);
    setScrapeTotalDuration(null);

    const startTime = Date.now();
    if (scrapeTimerRef.current) clearInterval(scrapeTimerRef.current);
    scrapeTimerRef.current = setInterval(() => {
      setScrapeElapsedSeconds(((Date.now() - startTime) / 1000).toFixed(1));
    }, 100);

    try {
      const options = {
        wait_seconds: parseFloat(singleWaitTime) || 2.0,
        use_stealth: singleUseStealth,
      };

      // First attempt with user selected mode (defaults to 'auto' for max resilience)
      let data = await api.scrapeWebPage(cleanUrl, scrapeMode || 'auto', options);
      
      // If user selected tables mode and no tables were found, automatically retry with 'auto' mode
      if ((!data || !data.active_table || !data.active_table.rows || data.active_table.rows.length === 0) && scrapeMode !== 'auto') {
        data = await api.scrapeWebPage(cleanUrl, 'auto', options);
      }

      if (scrapeTimerRef.current) clearInterval(scrapeTimerRef.current);

      if (data && data.active_table && data.active_table.rows && data.active_table.rows.length > 0) {
        setScrapedPreview(data.active_table);
        setScrapeTotalDuration(data.duration_seconds || ((Date.now() - startTime) / 1000).toFixed(2));
        setSingleScrapeError(null);
        setScrapedPreviewOpen(true);
        setScrapedPage(1);
        setScrapeLoading(false);
        return;
      }
    } catch (e) {
      console.error('Web scraping error:', e);
    }

    if (scrapeTimerRef.current) clearInterval(scrapeTimerRef.current);
    setScrapeLoading(false);
    const errMsg = isAr
      ? 'تعذر استخراج بيانات من هذا الرابط أو أن الصفحة تتطلب تسجيل دخول/مصادقة خاصة أو تعتمد كلياً على عارض جافاسكريبت معقد. جرب وضع "الاستخراج الذكي الشامل" وزيادة وقت الانتظار أو اختر أحد النماذج الجاهزة.'
      : 'Could not extract dataset rows from this URL. The page may require authentication or heavy JS rendering. Try increasing wait time or choose another link.';
    setSingleScrapeError(errMsg);
    showPlatformAlert(
      isAr ? 'تنبيه استخراج البيانات' : 'Scraping Diagnostics',
      errMsg,
      'error'
    );
  };

  const handleImportScraped = () => {
    if (scrapedPreview && scrapedPreview.rows && scrapedPreview.rows.length > 0) {
      const fileName = scrapedPreview.name || `WebScraped_${Date.now().toString().slice(-4)}.csv`;
      setDataset(scrapedPreview.rows, fileName, 'scrape', {
        url: scrapeType === 'multipage' ? (inferredPattern?.template || manualTemplate) : scrapeUrl,
        pages_count: multiScrapeStats?.length || 1,
      });
    }
  };

  // Multi-Page Pattern Scraper Handlers
  const handleInferPattern = async () => {
    setInferLoading(true);
    setInferError(null);
    setInferredPattern(null);

    // Method 3: Direct manual template with {page}
    if (patternMethod === 'template') {
      const tmpl = (manualTemplate || '').trim();
      if (!tmpl || !tmpl.includes('{page}')) {
        setInferError(isAr ? 'يجب أن يحتوي القالب على الكلمة المفتاحية {page} في موضع الرقم المتغير.' : 'Template must contain the {page} token.');
        setInferLoading(false);
        return;
      }
      const start = parseInt(crawlStartPage, 10);
      const startVal = isNaN(start) ? 1 : start;
      const step = parseInt(crawlStep, 10);
      const stepVal = isNaN(step) || step <= 0 ? 1 : step;

      const sample_urls = [
        tmpl.replace('{page}', String(startVal)),
        tmpl.replace('{page}', String(startVal + stepVal)),
        tmpl.replace('{page}', String(startVal + 2 * stepVal)),
      ];

      setInferredPattern({
        status: 'success',
        detected_param: 'manual_template',
        param_type: 'template',
        start_val: startVal,
        step: stepVal,
        template: tmpl,
        sample_urls: sample_urls,
        method: 'template',
        explanation_ar: `قالب يدوي معتمد: يبدأ من القيمة ${startVal} بمقدار خطوة ${stepVal}.`,
        explanation_en: `Manual template accepted: starts at ${startVal} with step ${stepVal}.`,
      });
      setInferLoading(false);
      return;
    }

    let payload = { mode: patternMethod };
    if (patternMethod === 'diff') {
      const u1 = cleanPastedUrl(diffUrl1);
      const u2 = cleanPastedUrl(diffUrl2);
      if (!u1) {
        setInferError(isAr ? 'يرجى إدخال رابط الصفحة الأولى للاستنتاج.' : 'Please enter at least the first URL.');
        setInferLoading(false);
        return;
      }
      payload.url1 = u1;
      payload.url2 = u2 || u1;
    } else if (patternMethod === 'auto') {
      const u1 = cleanPastedUrl(autoStartUrl);
      if (!u1) {
        setInferError(isAr ? 'يرجى إدخال رابط صفحة البداية.' : 'Please enter the starting URL.');
        setInferLoading(false);
        return;
      }
      payload.url1 = u1;
    }

    try {
      const res = await api.inferUrlPattern(payload);
      if (res && res.status === 'success') {
        setInferredPattern(res);
        if (res.start_val !== undefined) setCrawlStartPage(res.start_val);
        if (res.step !== undefined) setCrawlStep(res.step);
      } else {
        // Smart client fallback for single URL or two URLs if server didn't connect
        if (payload.url1) {
          const u1 = payload.url1;
          const matchParam = u1.match(/(?:[?&])(page|p|pg|offset|start|skip)=(\d+)/i);
          if (matchParam) {
            const pName = matchParam[1];
            const pVal = parseInt(matchParam[2], 10);
            const stepVal = pName.toLowerCase() === 'offset' || pName.toLowerCase() === 'skip' ? 25 : 1;
            const template = u1.replace(new RegExp(`([?&]${pName}=)${pVal}`), `$1{page}`);
            const sample_urls = [
              template.replace('{page}', String(pVal)),
              template.replace('{page}', String(pVal + stepVal)),
              template.replace('{page}', String(pVal + 2 * stepVal)),
            ];
            setInferredPattern({
              status: 'success',
              detected_param: pName,
              param_type: 'query_param_smart',
              start_val: pVal,
              step: stepVal,
              template: template,
              sample_urls: sample_urls,
              method: 'single_url_heuristic',
              explanation_ar: `تم التعرف الذكي على معامل الترقيم '${pName}=${pVal}' واشتقاق قالب السحب المتعدد.`,
              explanation_en: `Intelligently detected pagination parameter '${pName}=${pVal}' in URL.`,
            });
            setCrawlStartPage(pVal);
            setCrawlStep(stepVal);
            setInferLoading(false);
            return;
          }
        }
        setInferError(res?.message || (isAr ? 'تعذر استنتاج النمط. تحقق من صحة الروابط.' : 'Could not infer pattern.'));
      }
    } catch (e) {
      setInferError(e.message || (isAr ? 'حدث خطأ في الاتصال بالخادم.' : 'Server connection error.'));
    }
    setInferLoading(false);
  };

  const handleExecuteMultiScrape = async () => {
    const tmpl = inferredPattern?.template || (patternMethod === 'template' ? manualTemplate : null);
    if (!tmpl || !tmpl.includes('{page}')) {
      const missingMsg = isAr ? 'يرجى استنتاج النمط أولاً أو إدخال قالب يحتوي على المتغير {page}.' : 'Please infer a pattern or provide a template with {page}.';
      setMultiScrapeError(missingMsg);
      showPlatformAlert(
        isAr ? 'قالب الرابط غير محدد' : 'Missing URL Template',
        missingMsg,
        'warning'
      );
      return;
    }

    setMultiScrapeLoading(true);
    setMultiScrapeStats(null);
    setMultiScrapeError(null);
    setScrapeElapsedSeconds(0);
    setScrapeTotalDuration(null);

    const startTime = Date.now();
    if (scrapeTimerRef.current) clearInterval(scrapeTimerRef.current);
    scrapeTimerRef.current = setInterval(() => {
      setScrapeElapsedSeconds(((Date.now() - startTime) / 1000).toFixed(1));
    }, 100);

    try {
      const payload = {
        template_url: tmpl,
        start_page: parseInt(crawlStartPage, 10) || 1,
        num_pages: Math.min(Math.max(parseInt(crawlNumPages, 10) || 5, 1), 25),
        step: parseInt(crawlStep, 10) || 1,
        extract_mode: crawlExtractMode,
        delay_seconds: parseFloat(crawlDelay) || 1.5,
        wait_seconds: parseFloat(crawlWaitTime) || 2.0,
        use_stealth: crawlUseStealth,
        enable_jitter: crawlEnableJitter,
      };

      const data = await api.scrapeMultiPage(payload);
      if (scrapeTimerRef.current) clearInterval(scrapeTimerRef.current);

      if (data && data.status === 'success' && data.rows && data.rows.length > 0) {
        setScrapedPreview({
          table_index: 1,
          columns: data.columns,
          rows: data.rows,
          rows_count: data.rows.length,
          name: data.filename,
          total_duration_seconds: data.total_duration_seconds,
        });
        setScrapeTotalDuration(data.total_duration_seconds || ((Date.now() - startTime) / 1000).toFixed(2));
        setMultiScrapeStats(data.page_stats);
        setMultiScrapeError(null);
        setScrapedPreviewOpen(true);
        setScrapedPage(1);
        setMultiScrapeLoading(false);
        return;
      } else if (data && data.status === 'error') {
        const err = data.message || data.error || (isAr ? 'فشلت عملية السحب المتعدد من الخادم.' : 'Multi-page scraping failed on server.');
        setMultiScrapeError(err);
        showPlatformAlert(
          isAr ? 'فشل عملية السحب المتعدد' : 'Multi-Page Scraping Error',
          err,
          'error'
        );
      } else if (data && data.page_stats && (!data.rows || data.rows.length === 0)) {
        setMultiScrapeStats(data.page_stats);
        setScrapeTotalDuration(data.total_duration_seconds || ((Date.now() - startTime) / 1000).toFixed(2));
        const warn = data.message || (isAr ? 'تمت زيارة الصفحات بنجاح، ولكن لم يتم العثور على جداول أو عناصر منظمة في الصفحات المستهدفة. جرب وضع (كشف تلقائي شامل) وزيادة وقت انتظار تصيير العناصر.' : 'Pages visited, but no structured tables or records were found.');
        setMultiScrapeError(warn);
        showPlatformAlert(
          isAr ? 'تقرير السحب المتعدد: لا توجد بيانات' : 'Multi-Page Crawling Report: No Data',
          warn,
          'warning'
        );
      } else {
        const fallbackErr = data?.error || (isAr ? 'تعذر استلام بيانات من الخادم أو انتهت مهلة الاتصال.' : 'No data received from server or request timed out.');
        setMultiScrapeError(fallbackErr);
        showPlatformAlert(
          isAr ? 'خطأ في عملية السحب' : 'Scraping Error',
          fallbackErr,
          'error'
        );
      }
    } catch (e) {
      if (scrapeTimerRef.current) clearInterval(scrapeTimerRef.current);
      console.error('Multi-page scraping error:', e);
      const err = e.message || (isAr ? 'حدث خطأ غير متوقع أثناء الاتصال بالصفحات وسحب البيانات.' : 'Error during multi-page scraping execution.');
      setMultiScrapeError(err);
      showPlatformAlert(
        isAr ? 'خطأ في عملية السحب' : 'Scraping Error',
        err,
        'error'
      );
    }
    setMultiScrapeLoading(false);
  };

  // 3. Web Search & Pre-Fetch Handlers
  const handleSearch = async () => {
    if (!searchQuery.trim()) return;
    setSearchLoading(true);
    setSearchError(null);
    setSearchPage(1);
    setPreviewingDataset(null);
    setRemotePreviewResult(null);

    const userinput_n = Math.min(Math.max(parseInt(searchLimit, 10) || 20, 1), 50);

    try {
      const data = await api.searchWeb(searchQuery, userinput_n, searchPlatformFilters);
      if (data && data.results && data.results.length > 0) {
        setSearchResults(data.results);
        if (data.errors && data.errors.length > 0) {
          setSearchError(data.errors.join(' | '));
        }
        setSearchLoading(false);
        return;
      } else if (data && data.error) {
        setSearchError(data.error);
      } else if (data && data.message) {
        setSearchError(data.message);
      }
    } catch (e) {
      console.error('Search error:', e);
      setSearchError(e.message || (isAr ? 'حدث خطأ أثناء إجراء البحث.' : 'Search error occurred.'));
    }

    setSearchResults([]);
    setSearchLoading(false);
  };

  const handleFetchSamplePreview = async (item) => {
    setPreviewingDataset(item);
    setRemotePreviewResult(null);
    setFetchingRemote(true);

    try {
      // 1. Direct client-side fetch ONLY for verified CORS-safe raw GitHub or local URLs
      const isCorsSafe = item.download_url && (
        item.download_url.includes('raw.githubusercontent.com') ||
        item.download_url.startsWith('/') ||
        item.download_url.includes('localhost')
      );
      if (isCorsSafe) {
        try {
          const rawResp = await fetch(item.download_url);
          if (rawResp.ok) {
            const rawText = await rawResp.text();
            const parsed = parseRawTabularData(rawText, item.download_url);
            if (parsed && parsed.rows && parsed.rows.length > 0) {
              setRemotePreviewResult(parsed);
              setFetchingRemote(false);
              return;
            }
          }
        } catch (clientErr) {
          console.warn('Direct browser fetch fallback to server:', clientErr);
        }
      }

      // 2. Server-side fetch through FastAPI (handles Hugging Face API, GitHub API, proxies, encoding)
      const targetUrl = item.download_url || item.external_url;
      if (targetUrl) {
        const data = await api.fetchRemoteDataset(targetUrl);
        if (data && data.status === 'success' && data.rows && data.rows.length > 0) {
          setRemotePreviewResult(data);
          setFetchingRemote(false);
          return;
        }

        if (data && (data.status === 'search_portal' || data.is_search_portal)) {
          setRemotePreviewResult(data);
          setFetchingRemote(false);
          return;
        }

        if (data && data.message) {
          setRemotePreviewResult({
            status: 'error',
            message: data.message,
            external_url: item.external_url || item.download_url,
          });
          setFetchingRemote(false);
          return;
        }
      }
    } catch (e) {
      console.warn('Remote download error:', e);
    }

    setRemotePreviewResult({
      status: 'error',
      message: isAr
        ? 'تعذر قراءة أو استخراج جدول البيانات آلياً من الرابط الخارجي (قد يتطلب المستودع مصادقة خاصة أو تسجيلاً). يمكنك الانتقال للرابط الأصلي وتحميل الملف مباشرة ثم رفعه يدوياً.'
        : 'Could not stream table sample directly from the external link. You can visit the repository page directly or upload the file manually.',
      external_url: item.external_url || item.download_url,
    });
    setFetchingRemote(false);
  };

  const handleConfirmRemoteImport = () => {
    if (remotePreviewResult && remotePreviewResult.rows && remotePreviewResult.rows.length > 0) {
      const cleanTitle = (previewingDataset?.title || 'RemoteDataset')
        .replace(/[\(\)\[\]\{\}]/g, '')
        .trim()
        .slice(0, 40);
      const filename = remotePreviewResult.filename || `${cleanTitle}.csv`;
      setDataset(
        remotePreviewResult.rows,
        filename,
        'search',
        { url: previewingDataset?.download_url || previewingDataset?.external_url }
      );
      setPreviewingDataset(null);
      setRemotePreviewResult(null);
      setSearchSuccessMsg(
        isAr
          ? `تم تحميل واستيراد "${filename}" بنجاح (${remotePreviewResult.rows.length} صفاً، ${remotePreviewResult.columns?.length || Object.keys(remotePreviewResult.rows[0] || {}).length} عموداً) إلى بيئة العمل الحالية!`
          : `Successfully imported "${filename}" (${remotePreviewResult.rows.length} rows, ${remotePreviewResult.columns?.length || Object.keys(remotePreviewResult.rows[0] || {}).length} cols) into the workspace!`
      );
      setTimeout(() => setSearchSuccessMsg(null), 8000);
    }
  };

  const handleInstantSynthesizeSample = () => {
    const topic = previewingDataset?.title || searchQuery || 'Domain Dataset';
    const { data, filename } = generateDomainAcademicSample(topic, 60);
    setDataset(
      data,
      filename,
      'search',
      { url: previewingDataset?.external_url || 'in-platform-synthetic' }
    );
    setPreviewingDataset(null);
    setRemotePreviewResult(null);
    setSearchSuccessMsg(
      isAr
        ? `تم توليد واستيراد عينة أكاديمية معيارية لـ "${filename}" بنجاح (${data.length} صفاً) داخل المنصة مباشرة!`
        : `Successfully generated and imported academic benchmark sample "${filename}" (${data.length} rows) inside the Studio!`
    );
    setTimeout(() => setSearchSuccessMsg(null), 8000);
  };

  // 4. Quick Presets Generator
  const handleGeneratePreset = () => {
    let data = [];
    let filename = '';
    if (syntheticType === 'sentiment') {
      data = generateArabicSentiment(syntheticCount);
      filename = `Synthetic_Arabic_Sentiment_${syntheticCount}.csv`;
    } else if (syntheticType === 'news') {
      data = generateArabicNews(syntheticCount);
      filename = `Synthetic_Arabic_News_${syntheticCount}.csv`;
    } else if (syntheticType === 'iris') {
      data = generateIrisData();
      filename = 'Synthetic_Iris_Flowers.csv';
    } else if (syntheticType === 'titanic') {
      data = generateTitanicData(syntheticCount);
      filename = 'Synthetic_Titanic_Demographics.csv';
    }

    setDataset(data, filename, 'synthetic');
  };

  // Guided Custom Builder Handlers
  const handleAddCustomCol = () => {
    setCustomColumns([
      ...customColumns,
      { id: Date.now(), name: `col_${customColumns.length + 1}`, type: 'numeric', min: 0, max: 100 },
    ]);
  };

  const handleRemoveCustomCol = (id) => {
    if (customColumns.length <= 1) return;
    setCustomColumns(customColumns.filter((c) => c.id !== id));
  };

  const handleUpdateCustomCol = (id, key, val) => {
    setCustomColumns(
      customColumns.map((c) => (c.id === id ? { ...c, [key]: val } : c))
    );
  };

  const handleGenerateCustomGuided = () => {
    const arabicTexts = [
      'معمارية المحول رائعة وأظهرت استقراراً مبهراً في التدريب.',
      'دقة النموذج تجاوزت التوقعات مع انخفاض ملحوظ في دالة الخسارة.',
      'خوارزمية RoPE حسنت استيعاب المسافات الطويلة بين الكلمات بوضوح.',
      'تطبيق درع EWC منع النسيان الكارثي وحافظ على أداء المهام السابقة.',
      'أداء ممتاز وسرعة معالجة عالية في استخلاص الخصائص اللغوية.',
      'حدث تذبذب شديد في التدرجات وانفجار في الأوزان أثناء التعلم.',
      'انخفضت الدقة على المهمة القديمة بسبب ظاهرة النسيان الكارثي.',
      'البيانات تحوي قيماً شاذة كثيرة أدت لتشتت دالة التكلفة.',
    ];

    const rows = [];
    for (let i = 0; i < customRowsCount; i++) {
      const row = { id: i + 1 };
      customColumns.forEach((col) => {
        if (col.type === 'arabic_text') {
          row[col.name] = arabicTexts[Math.floor(Math.random() * arabicTexts.length)];
        } else if (col.type === 'category') {
          const cats = col.categories
            ? col.categories.split(',').map((c) => c.trim()).filter(Boolean)
            : ['Class_A', 'Class_B'];
          row[col.name] = cats[Math.floor(Math.random() * cats.length)] || 'Class_A';
        } else if (col.type === 'numeric') {
          const min = Number(col.min) || 0;
          const max = Number(col.max) || 100;
          row[col.name] = +(min + Math.random() * (max - min)).toFixed(2);
        } else if (col.type === 'date') {
          const month = String(Math.floor(Math.random() * 12) + 1).padStart(2, '0');
          const day = String(Math.floor(Math.random() * 28) + 1).padStart(2, '0');
          row[col.name] = `2025-${month}-${day}`;
        }
      });
      rows.push(row);
    }

    setDataset(rows, customDatasetName.trim() || 'Custom_Synthetic_Corpus.csv', 'synthetic');
  };

  // Manual Matrix Entry
  const handleImportManual = () => {
    try {
      const lines = manualText.split(/\r?\n/).filter((l) => l.trim().length > 0);
      if (lines.length < 2) {
        showPlatformAlert(
          isAr ? 'خطأ في الإدخال اليدوي' : 'Manual Entry Error',
          isAr ? 'يرجى إدخال سطر ترويسة وسطر بيانات واحد على الأقل.' : 'Enter at least a header row and one data row.',
          'warning'
        );
        return;
      }
      const headers = lines[0].split(',').map((h) => h.trim().replace(/^"|"$/g, ''));
      const rows = lines.slice(1).map((line) => {
        const values = line.split(',').map((v) => v.trim().replace(/^"|"$/g, ''));
        const obj = {};
        headers.forEach((h, i) => {
          obj[h || `col_${i + 1}`] = values[i] !== undefined ? values[i] : null;
        });
        return obj;
      });
      setDataset(rows, 'Manual_Matrix_Entry.csv', 'manual');
    } catch (err) {
      showPlatformAlert(
        isAr ? 'خطأ في التحليل' : 'Parse Error',
        isAr ? 'تعذر قراءة النص المدخل. تأكد من صحة التنسيق وفواصل الأعمدة.' : 'Failed to parse manual matrix text. Verify format and delimiters.',
        'error'
      );
    }
  };

  // Direct Dataset Downloader helper for Sample Benchmark cards
  const handleDownloadSampleCSV = (item) => {
    let data = [];
    if (item.id === 'arabic_sentiment') data = generateArabicSentiment(60);
    else if (item.id === 'arabic_news_classification') data = generateArabicNews(50);
    else if (item.id === 'iris_flowers') data = generateIrisData();
    else if (item.id === 'titanic_survival') data = generateTitanicData(70);
    else data = generateArabicSentiment(50);

    const headers = Object.keys(data[0] || {});
    const rows = data.map((row) =>
      headers
        .map((h) => {
          const val = row[h];
          if (val === null || val === undefined) return '';
          const str = String(val);
          return str.includes(',') || str.includes('\n') || str.includes('"')
            ? `"${str.replace(/"/g, '""')}"`
            : str;
        })
        .join(',')
    );
    const csvContent = '\uFEFF' + [headers.join(','), ...rows].join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${item.id}.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const sourceTabs = [
    { id: 'upload', code: '01.1', title: isAr ? 'رفع ملف محلي' : 'File Upload', badge: 'CSV / JSON' },
    { id: 'scrape', code: '01.2', title: isAr ? 'استخراج من الويب (Web Scraping)' : 'Web Scraping', badge: 'URL / Table' },
    { id: 'search', code: '01.3', title: isAr ? 'بحث في الإنترنت' : 'Web Search', badge: 'Live Queries' },
    { id: 'synthetic', code: '01.4', title: isAr ? 'توليد بيانات اصطناعية' : 'Synthetic Data', badge: 'NLP & Custom' },
    { id: 'samples', code: '01.5', title: isAr ? 'عينات أكاديمية جاهزة' : 'Classic Benchmarks', badge: 'Curated' },
    { id: 'library', code: '01.6', title: isAr ? 'المجموعات المحفوظة في مكتبة الاستوديو' : 'Saved Studio Library', badge: `${savedLibrary?.length || 0} ${isAr ? 'مجموعات' : 'saved'}` },
  ];

  const formatLibraryDate = (rawDate) => {
    if (!rawDate) return isAr ? 'محفوظة' : 'Saved';
    try {
      const d = new Date(rawDate);
      if (isNaN(d.getTime())) {
        return String(rawDate).replace('T', ' ').slice(0, 16);
      }
      const yyyy = d.getFullYear();
      const mm = String(d.getMonth() + 1).padStart(2, '0');
      const dd = String(d.getDate()).padStart(2, '0');
      const hh = String(d.getHours()).padStart(2, '0');
      const min = String(d.getMinutes()).padStart(2, '0');
      return `${yyyy}-${mm}-${dd} ${hh}:${min}`;
    } catch {
      return String(rawDate).slice(0, 10);
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      {/* Introduction Banner & Pedagogical Guidance */}
      <div
        style={{
          background: 'var(--bg-secondary)',
          border: '1px solid var(--border-subtle)',
          borderRadius: 'var(--radius-lg, 12px)',
          padding: '1.25rem 1.5rem',
          boxShadow: 'var(--shadow-sm)',
        }}
      >
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '1rem' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.4rem' }}>
              <span className="badge badge-teal">{isAr ? 'المرحلة 1 من 5' : 'Phase 1 of 5'}</span>
              <span style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>
                <TermTooltip termKey="acquisition" customLabel={isAr ? 'منهجية CRISP-DM: جلب البيانات' : 'CRISP-DM: Data Acquisition'} />
              </span>
            </div>
            <p style={{ margin: 0, fontSize: '0.9rem', color: 'var(--text-secondary)', lineHeight: 1.6, maxWidth: '850px' }}>
              {isAr ? (
                <>
                  <TermTooltip termKey="acquisition" customLabel="جلب البيانات (Data Acquisition)" /> هو نقطة البداية الأساسية: بدون بيانات جيدة لا يمكن لأي نموذج عصبي أن يتعلم بشكل صحيح (<TermTooltip termKey="gigo" customLabel="مبدأ GIGO: Garbage In, Garbage Out" />). اختر المصدر الأنسب لمهمتك من الوسائط الخمس أدناه.
                </>
              ) : (
                <>
                  <TermTooltip termKey="acquisition" customLabel="Data Acquisition" /> initiates the empirical machine learning lifecycle. Under <TermTooltip termKey="gigo" customLabel="GIGO Principle" />, choose from 5 ingestion modalities below.
                </>
              )}
            </p>
          </div>

          <button
            onClick={() => setShowHelper(!showHelper)}
            className="btn btn-secondary btn-sm"
            style={{ fontSize: '0.78rem', gap: '0.4rem' }}
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="10"></circle>
              <line x1="12" y1="16" x2="12" y2="12"></line>
              <line x1="12" y1="8" x2="12.01" y2="8"></line>
            </svg>
            <span>{showHelper ? (isAr ? 'إخفاء دليل الاختيار' : 'Hide Helper') : (isAr ? 'كيف أختار المصدر المناسب؟' : 'How to Choose?')}</span>
          </button>
        </div>

        {/* Expandable Decision Helper */}
        {showHelper && (
          <div
            style={{
              marginTop: '1rem',
              paddingTop: '1rem',
              borderTop: '1px solid var(--border-subtle)',
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
              gap: '0.75rem',
            }}
          >
            {[
              { title_ar: 'رفع ملف محلي', title_en: 'File Upload', text_ar: 'عندما تملك بالفعل ملف CSV أو JSON مسحوباً من قاعدة بياناتك أو نظامك التجاري.', text_en: 'When you already possess private domain data.' },
              { title_ar: 'استخراج من الويب (Web Scraping)', title_en: 'Web Scraping', text_ar: 'عندما تجد جدول بيانات منشوراً في صفحة ويكيبيديا أو موقع حكومي عام وتريد سحبه.', text_en: 'Extract tabular matrices from public web pages.' },
              { title_ar: 'بحث في الإنترنت', title_en: 'Web Search', text_ar: 'للبحث المباشر عن أحدث الأوراق والمقاييس وتحويل نتائج البحث لجدول منظم.', text_en: 'Search live scientific repositories and convert findings.' },
              { title_ar: 'توليد اصطناعي', title_en: 'Synthetic Data', text_ar: 'مثالي للتدريب الأولي، واختبار المعمارية، وفهم آليات التدريب قبل جمع البيانات الحقيقية.', text_en: 'Ideal for prototyping and understanding model mechanics.' },
              { title_ar: 'عينات جاهزة', title_en: 'Curated Benchmarks', text_ar: 'لمقارنة نموذجك مع معايير كلاسيكية معتمدة عالمياً (مثل مشاعر اللغة العربية أو الأخبار).', text_en: 'Standard benchmarks for verifiable scientific comparison.' },
            ].map((guide, idx) => (
              <div key={idx} style={{ background: 'var(--bg-tertiary)', padding: '0.75rem', borderRadius: 'var(--radius-sm, 6px)', border: '1px solid var(--border-subtle)' }}>
                <div style={{ fontWeight: '800', fontSize: '0.82rem', color: 'var(--accent-teal)', marginBottom: '0.25rem' }}>
                  {isAr ? guide.title_ar : guide.title_en}
                </div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', lineHeight: 1.45 }}>
                  {isAr ? guide.text_ar : guide.text_en}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Source Selector Navigation */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(170px, 1fr))',
          gap: '0.75rem',
        }}
      >
        {sourceTabs.map((tab) => {
          const isActive = activeSource === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveSource(tab.id)}
              style={{
                background: isActive ? 'var(--accent-teal-glow)' : 'var(--bg-secondary)',
                border: isActive ? '1px solid var(--accent-teal)' : '1px solid var(--border-subtle)',
                borderRadius: 'var(--radius-md, 8px)',
                padding: '0.85rem 1rem',
                textAlign: isAr ? 'right' : 'left',
                cursor: 'pointer',
                transition: 'all 0.15s ease',
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.25rem' }}>
                <span style={{ fontSize: '0.7rem', fontFamily: 'var(--font-mono)', color: 'var(--accent-teal)', fontWeight: '700' }}>
                  [{tab.code}]
                </span>
                <span className={`badge ${isActive ? 'badge-teal' : 'badge-neutral'}`} style={{ fontSize: '0.65rem' }}>
                  {tab.badge}
                </span>
              </div>
              <div style={{ fontSize: '0.88rem', fontWeight: 800, color: isActive ? 'var(--accent-teal)' : 'var(--text-primary)' }}>
                {tab.title}
              </div>
            </button>
          );
        })}
      </div>

      {/* Source 1: Local File Upload */}
      {activeSource === 'upload' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
          <div
            onDragOver={(e) => {
              e.preventDefault();
              setIsDragging(true);
            }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={(e) => {
              e.preventDefault();
              setIsDragging(false);
              const file = e.dataTransfer.files?.[0];
              if (file) processUploadedFile(file);
            }}
            style={{
              border: isDragging ? '2px dashed var(--accent-teal)' : '2px dashed var(--border-medium)',
              borderRadius: 'var(--radius-lg, 12px)',
              padding: '3rem 2rem',
              textAlign: 'center',
              background: isDragging ? 'var(--accent-teal-glow)' : 'var(--bg-secondary)',
              transition: 'all 0.2s ease',
              cursor: 'pointer',
            }}
          >
            <div
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                justifyContent: 'center',
                width: '56px',
                height: '56px',
                borderRadius: '12px',
                background: 'var(--accent-teal-glow)',
                border: '1px solid var(--accent-teal)',
                marginBottom: '0.85rem',
                color: 'var(--accent-teal)',
              }}
            >
              <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                <polyline points="17 8 12 3 7 8"></polyline>
                <line x1="12" y1="3" x2="12" y2="15"></line>
              </svg>
            </div>

            <h3 style={{ fontSize: '1.15rem', fontWeight: 800, color: 'var(--text-primary)', marginBottom: '0.4rem' }}>
              {isAr ? 'اسحب وأفلت ملف البيانات هنا أو تصفح جهازك' : 'Drag & Drop Your Data File Here or Browse'}
            </h3>

            <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '1.25rem' }}>
              {isAr ? 'الصيغ المدعومة: CSV (.csv), JSON (.json), TSV (.tsv) مع معالجة مباشرة داخل المتصفح' : 'Supported formats: CSV, JSON, TSV with zero latency client parsing'}
            </p>

            <label
              className="btn btn-primary"
              style={{ cursor: 'pointer', padding: '0.65rem 1.75rem', fontSize: '0.88rem', fontWeight: 800 }}
            >
              <span>{isAr ? 'استعراض واختيار ملف من جهازك' : 'Browse Local Files'}</span>
              <input type="file" accept=".csv,.json,.tsv,.txt" onChange={handleFileUpload} style={{ display: 'none' }} />
            </label>

            {uploadError && (
              <div style={{ marginTop: '1rem', color: 'var(--accent-rose)', fontSize: '0.85rem', fontWeight: '700' }}>
                {uploadError}
              </div>
            )}
          </div>

          {/* Uploaded File Pre-Confirmation Preview */}
          {uploadedPreview && (
            <div
              style={{
                background: 'var(--bg-secondary)',
                border: '1px solid var(--accent-teal)',
                borderRadius: 'var(--radius-lg, 12px)',
                padding: '1.25rem 1.5rem',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                flexWrap: 'wrap',
                gap: '1rem',
              }}
            >
              <div>
                <span className="badge badge-teal" style={{ marginBottom: '0.35rem' }}>
                  {isAr ? 'تم التحليل بنجاح' : 'Parsed Successfully'}
                </span>
                <h4 style={{ margin: 0, fontSize: '1rem', fontWeight: 800, color: 'var(--text-primary)', fontFamily: 'var(--font-mono)' }}>
                  {uploadedPreview.name}
                </h4>
                <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '0.25rem' }}>
                  {uploadedPreview.rowsCount} {isAr ? 'صفاً' : 'rows'} · {uploadedPreview.colsCount} {isAr ? 'أعمدة' : 'columns'} · {Math.round(uploadedPreview.size / 1024)} KB
                </div>
              </div>

              <div style={{ display: 'flex', gap: '0.5rem' }}>
                <button
                  onClick={() => setUploadedPreview(null)}
                  className="btn btn-secondary btn-sm"
                >
                  {isAr ? 'إلغاء' : 'Cancel'}
                </button>
                <button
                  onClick={handleConfirmUpload}
                  className="btn btn-primary btn-sm"
                  style={{ fontWeight: 800, display: 'inline-flex', alignItems: 'center', gap: '0.45rem' }}
                >
                  <span>{isAr ? 'اعتماد وإدخال في الاستوديو' : 'Confirm & Load Dataset'}</span>
                  <ArrowRight size={14} style={{ transform: isAr ? 'scaleX(-1)' : 'none' }} />
                </button>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Source 2: Web Scraping */}
      {activeSource === 'scrape' && (
        <div
          style={{
            background: 'var(--bg-secondary)',
            border: '1px solid var(--border-subtle)',
            borderRadius: 'var(--radius-lg, 12px)',
            padding: '1.5rem',
            display: 'flex',
            flexDirection: 'column',
            gap: '1.25rem',
          }}
        >
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.75rem' }}>
            <div>
              <h3 style={{ fontSize: '1.1rem', fontWeight: 800, color: 'var(--text-primary)', marginBottom: '0.35rem' }}>
                <TermTooltip termKey="web_scraping" customLabel={isAr ? 'سحب البيانات من صفحات الويب (Web Scraping Studio)' : 'Web Data Scraping & Crawling'} />
              </h3>
              <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', margin: 0 }}>
                {isAr
                  ? 'اسحب الجداول والبيانات من صفحة واحدة، أو استخدم الزاحف الذكي لاستنتاج نمط الروابط والسحب المتعدد عبر عدة صفحات آلياً.'
                  : 'Scrape data from a single page, or unleash the multi-page pattern crawler to harvest continuous records automatically.'}
              </p>
            </div>

            {/* Mode Switcher: Single Page vs Multi-Page Crawler */}
            <div style={{ display: 'inline-flex', padding: '0.25rem', background: 'var(--bg-tertiary)', borderRadius: 'var(--radius-md, 8px)', border: '1px solid var(--border-subtle)', gap: '0.25rem' }}>
              <button
                onClick={() => setScrapeType('multipage')}
                style={{
                  padding: '0.45rem 1rem',
                  borderRadius: 'var(--radius-sm, 6px)',
                  border: 'none',
                  background: scrapeType === 'multipage' ? 'var(--accent-teal)' : 'transparent',
                  color: scrapeType === 'multipage' ? '#ffffff' : 'var(--text-secondary)',
                  fontWeight: '800',
                  fontSize: '0.82rem',
                  cursor: 'pointer',
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: '0.45rem',
                  transition: 'all 0.15s ease',
                }}
              >
                <Layers size={14} />
                <span>{isAr ? 'سحب متعدد عبر نمط الروابط' : 'Multi-Page Pattern Crawler'}</span>
              </button>
              <button
                onClick={() => setScrapeType('single')}
                style={{
                  padding: '0.45rem 1rem',
                  borderRadius: 'var(--radius-sm, 6px)',
                  border: 'none',
                  background: scrapeType === 'single' ? 'var(--accent-teal)' : 'transparent',
                  color: scrapeType === 'single' ? '#ffffff' : 'var(--text-secondary)',
                  fontWeight: '800',
                  fontSize: '0.82rem',
                  cursor: 'pointer',
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: '0.45rem',
                  transition: 'all 0.15s ease',
                }}
              >
                <Globe size={14} />
                <span>{isAr ? 'صفحة واحدة فقط' : 'Single Page'}</span>
              </button>
            </div>
          </div>

          {/* SUB-VIEW 1: Single Page Mode */}
          {scrapeType === 'single' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
              {/* Quick Single-Page Presets Bar */}
              <div style={{ background: 'var(--bg-tertiary)', padding: '0.85rem 1rem', borderRadius: 'var(--radius-md, 8px)', border: '1px solid var(--border-subtle)', display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.5rem' }}>
                  <span style={{ fontSize: '0.78rem', fontWeight: 800, color: 'var(--text-primary)', display: 'inline-flex', alignItems: 'center', gap: '0.35rem' }}>
                    <Sparkles size={13} color="var(--accent-teal)" />
                    <span>{isAr ? 'نماذج حية جاهزة للاستخراج المباشر بنقرة واحدة:' : 'Curated Single-Page Live Presets:'}</span>
                  </span>
                  <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
                    {isAr ? 'اضغط لتعبئة الرابط واختبار السحب الفوري' : 'Click to populate and test instant extraction'}
                  </span>
                </div>

                <div style={{ display: 'flex', gap: '0.45rem', flexWrap: 'wrap' }}>
                  {singlePagePresets.map((pr) => (
                    <button
                      key={pr.id}
                      type="button"
                      onClick={() => setScrapeUrl(pr.url)}
                      className="btn btn-ghost btn-sm"
                      style={{
                        fontSize: '0.76rem',
                        padding: '0.35rem 0.65rem',
                        borderRadius: 'var(--radius-sm, 6px)',
                        border: '1px solid var(--border-medium)',
                        background: 'var(--bg-secondary)',
                        color: 'var(--text-primary)',
                        display: 'inline-flex',
                        alignItems: 'center',
                        gap: '0.35rem',
                        transition: 'all 0.15s ease',
                      }}
                    >
                      <span className="badge badge-teal" style={{ fontSize: '0.65rem', padding: '0.1rem 0.35rem' }}>
                        {pr.badge}
                      </span>
                      <span style={{ fontWeight: 700 }}>{isAr ? pr.name_ar : pr.name_en}</span>
                    </button>
                  ))}
                </div>
              </div>

              {/* Input Form Controls */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap' }}>
                  <input
                    type="url"
                    value={scrapeUrl}
                    onChange={(e) => setScrapeUrl(e.target.value)}
                    placeholder="https://arxiv.org/list/cs.AI/recent"
                    dir="ltr"
                    style={{
                      flex: 1,
                      minWidth: '280px',
                      padding: '0.65rem 1rem',
                      background: 'var(--bg-tertiary)',
                      border: '1px solid var(--border-medium)',
                      borderRadius: 'var(--radius-sm, 6px)',
                      color: 'var(--text-primary)',
                      fontFamily: 'var(--font-mono)',
                      fontSize: '0.85rem',
                      textAlign: 'left',
                      direction: 'ltr',
                    }}
                  />

                  <select
                    value={scrapeMode}
                    onChange={(e) => setScrapeMode(e.target.value)}
                    style={{
                      padding: '0.65rem 1rem',
                      background: 'var(--bg-tertiary)',
                      border: '1px solid var(--border-medium)',
                      borderRadius: 'var(--radius-sm, 6px)',
                      color: 'var(--text-primary)',
                      fontSize: '0.85rem',
                    }}
                  >
                    <option value="auto">{isAr ? 'الاستخراج الذكي الشامل (تلقائي)' : 'Auto-Extract (All Structures)'}</option>
                    <option value="tables">{isAr ? 'استخراج الجداول وقوائم HTML' : 'Extract Tables Only'}</option>
                    <option value="cards">{isAr ? 'بطاقات وعناصر مكررة (Cards / Items)' : 'Cards / Product Items'}</option>
                    <option value="text">{isAr ? 'استخراج المقالات والنصوص الكاملة' : 'Extract Paragraphs & Text'}</option>
                  </select>

                  <button
                    onClick={handleScrape}
                    disabled={scrapeLoading || !scrapeUrl.trim()}
                    className="btn btn-primary"
                    style={{ padding: '0.65rem 1.75rem', fontWeight: 800, display: 'inline-flex', alignItems: 'center', gap: '0.5rem' }}
                  >
                    <Play size={14} />
                    <span>{scrapeLoading ? (isAr ? `جاري الاستخراج... (${scrapeElapsedSeconds} ث)` : `Scraping... (${scrapeElapsedSeconds}s)`) : (isAr ? 'استخراج البيانات' : 'Scrape Web')}</span>
                  </button>
                </div>

                {/* Timing and Stealth Options for Single Page */}
                <div style={{ display: 'flex', gap: '1.25rem', alignItems: 'center', flexWrap: 'wrap', padding: '0.5rem 0.85rem', background: 'var(--bg-tertiary)', borderRadius: 'var(--radius-sm, 6px)', border: '1px solid var(--border-subtle)', fontSize: '0.78rem' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.45rem' }}>
                    <span style={{ color: 'var(--text-secondary)', fontWeight: 700 }}>
                      {isAr ? 'انتظار تحميل العناصر الديناميكية (ثانية):' : 'Dynamic JS Wait (s):'}
                    </span>
                    <input
                      type="number"
                      step="0.5"
                      min="0.2"
                      max="10"
                      value={singleWaitTime}
                      onChange={(e) => setSingleWaitTime(e.target.value)}
                      style={{ width: '65px', padding: '0.25rem 0.45rem', background: 'var(--bg-secondary)', border: '1px solid var(--border-medium)', borderRadius: 'var(--radius-sm, 4px)', color: 'var(--text-primary)', fontSize: '0.78rem', textAlign: 'center' }}
                    />
                  </div>
                  <label style={{ display: 'inline-flex', alignItems: 'center', gap: '0.35rem', cursor: 'pointer', color: 'var(--text-secondary)' }}>
                    <input
                      type="checkbox"
                      checked={singleUseStealth}
                      onChange={(e) => setSingleUseStealth(e.target.checked)}
                      style={{ accentColor: 'var(--accent-teal)' }}
                    />
                    <span>{isAr ? 'تشغيل المتصفح الخفي (Headless Stealth)' : 'Enable Stealth Browser'}</span>
                  </label>
                </div>

                {/* Inline Single Page Error Banner */}
                {singleScrapeError && (
                  <div style={{ padding: '0.75rem 1rem', background: 'rgba(239, 68, 68, 0.12)', border: '1px solid var(--accent-rose, #ef4444)', borderRadius: 'var(--radius-sm, 6px)', color: 'var(--accent-rose, #ef4444)', fontSize: '0.82rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <AlertCircle size={16} style={{ flexShrink: 0 }} />
                    <span>{singleScrapeError}</span>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* SUB-VIEW 2: Advanced Multi-Page Pattern Crawler */}
          {scrapeType === 'multipage' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
              {/* Pattern Discovery Method Selector */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                <span style={{ fontSize: '0.82rem', fontWeight: 800, color: 'var(--text-primary)' }}>
                  {isAr ? 'اختر طريقة تحديد نمط الروابط وعلاقة الترقيم:' : 'Select URL Pattern Discovery Method:'}
                </span>
                <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
                  {[
                    { id: 'diff', icon: Sparkles, label_ar: 'الاستنتاج الآلي من رابطين (موصى به)', label_en: 'Two URLs Diff (Recommended)' },
                    { id: 'auto', icon: Search, label_ar: 'الفحص الآلي لصفحة البداية', label_en: 'Auto DOM Discovery' },
                    { id: 'template', icon: Sliders, label_ar: 'إدخال القالب يدوياً مع {page}', label_en: 'Manual {page} Template' },
                  ].map((m) => {
                    const MethodIcon = m.icon;
                    const isActive = patternMethod === m.id;
                    return (
                      <button
                        key={m.id}
                        onClick={() => {
                          setPatternMethod(m.id);
                          setInferredPattern(null);
                          setInferError(null);
                        }}
                        style={{
                          padding: '0.5rem 1rem',
                          borderRadius: 'var(--radius-sm, 6px)',
                          border: isActive ? '1px solid var(--accent-teal)' : '1px solid var(--border-medium)',
                          background: isActive ? 'rgba(20, 184, 166, 0.12)' : 'var(--bg-tertiary)',
                          color: isActive ? 'var(--accent-teal)' : 'var(--text-secondary)',
                          fontSize: '0.82rem',
                          fontWeight: isActive ? 800 : 600,
                          cursor: 'pointer',
                          display: 'inline-flex',
                          alignItems: 'center',
                          gap: '0.45rem',
                          transition: 'all 0.15s ease',
                        }}
                      >
                        <MethodIcon size={14} />
                        <span>{isAr ? m.label_ar : m.label_en}</span>
                      </button>
                    );
                  })}
                </div>
              </div>

              {/* Curated Multi-Page Presets Bar */}
              <div style={{ background: 'var(--bg-tertiary)', padding: '0.85rem 1rem', borderRadius: 'var(--radius-md, 8px)', border: '1px solid var(--border-subtle)', display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.5rem' }}>
                  <span style={{ fontSize: '0.78rem', fontWeight: 800, color: 'var(--text-primary)', display: 'inline-flex', alignItems: 'center', gap: '0.35rem' }}>
                    <Sparkles size={13} color="var(--accent-teal)" />
                    <span>{isAr ? 'أمثلة ونماذج جاهزة لصفحات ومستودعات علمية متعددة الترقيم:' : 'Curated Scientific & Academic Multi-Page Presets:'}</span>
                  </span>
                  <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
                    {isAr ? 'اضغط على أي نموذج لتعبئة الروابط ومعاينتها فورياً' : 'Click any preset to auto-load and verify instantly'}
                  </span>
                </div>

                <div style={{ display: 'flex', gap: '0.45rem', flexWrap: 'wrap' }}>
                  {multiPagePresets.map((pr) => (
                    <button
                      key={pr.id}
                      type="button"
                      onClick={() => applyMultiPagePreset(pr)}
                      className="btn btn-ghost btn-sm"
                      style={{
                        fontSize: '0.76rem',
                        padding: '0.35rem 0.65rem',
                        borderRadius: 'var(--radius-sm, 6px)',
                        border: '1px solid var(--border-medium)',
                        background: 'var(--bg-secondary)',
                        color: 'var(--text-primary)',
                        display: 'inline-flex',
                        alignItems: 'center',
                        gap: '0.35rem',
                        transition: 'all 0.15s ease',
                      }}
                    >
                      <span className="badge badge-teal" style={{ fontSize: '0.65rem', padding: '0.1rem 0.35rem' }}>
                        {pr.badge}
                      </span>
                      <span style={{ fontWeight: 700 }}>{isAr ? pr.name_ar : pr.name_en}</span>
                    </button>
                  ))}
                </div>
              </div>

              {/* Method 1: Two URLs Diff */}
              {patternMethod === 'diff' && (() => {
                const domain1 = getDomainFromUrl(diffUrl1);
                const domain2 = getDomainFromUrl(diffUrl2);
                const hasHostMismatch = domain1 && domain2 && domain1 !== domain2;

                return (
                  <div style={{ background: 'var(--bg-tertiary)', padding: '1rem 1.25rem', borderRadius: 'var(--radius-md, 8px)', border: '1px solid var(--border-subtle)', display: 'flex', flexDirection: 'column', gap: '0.85rem' }}>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '0.85rem' }}>
                      {/* URL 1 Input */}
                      <div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.35rem' }}>
                          <label style={{ fontSize: '0.78rem', fontWeight: 700, color: 'var(--text-secondary)' }}>
                            {isAr ? 'رابط الصفحة الأولى (e.g. skip=0 أو page=1):' : 'Page 1 URL:'}
                          </label>
                          {domain1 && (
                            <span style={{ fontSize: '0.7rem', color: 'var(--accent-teal)', fontFamily: 'var(--font-mono)' }}>
                              {domain1}
                            </span>
                          )}
                        </div>
                        <div style={{ position: 'relative', display: 'flex', alignItems: 'center' }}>
                          <input
                            type="url"
                            dir="ltr"
                            value={diffUrl1}
                            onChange={(e) => setDiffUrl1(cleanPastedUrl(e.target.value))}
                            placeholder="https://arxiv.org/list/cs.AI/recent?skip=0&show=25"
                            style={{
                              width: '100%',
                              padding: isAr ? '0.55rem 2rem 0.55rem 0.75rem' : '0.55rem 0.75rem 0.55rem 2rem',
                              background: 'var(--bg-secondary)',
                              border: '1px solid var(--border-medium)',
                              borderRadius: 'var(--radius-sm, 6px)',
                              color: 'var(--text-primary)',
                              fontFamily: 'var(--font-mono)',
                              fontSize: '0.82rem',
                              textAlign: 'left',
                              direction: 'ltr',
                            }}
                          />
                          {diffUrl1 && (
                            <button
                              type="button"
                              onClick={() => setDiffUrl1('')}
                              title={isAr ? 'مسح الرابط' : 'Clear URL'}
                              style={{
                                position: 'absolute',
                                [isAr ? 'right' : 'left']: '0.45rem',
                                background: 'transparent',
                                border: 'none',
                                color: 'var(--text-muted)',
                                cursor: 'pointer',
                                padding: '0.2rem',
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                              }}
                            >
                              <X size={14} />
                            </button>
                          )}
                        </div>
                      </div>

                      {/* URL 2 Input */}
                      <div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.35rem' }}>
                          <label style={{ fontSize: '0.78rem', fontWeight: 700, color: 'var(--text-secondary)' }}>
                            {isAr ? 'رابط الصفحة الثانية لنفس الموقع (e.g. page=2):' : 'Page 2 URL (same site):'}
                          </label>
                          {domain2 && (
                            <span style={{ fontSize: '0.7rem', color: hasHostMismatch ? 'var(--accent-amber)' : 'var(--accent-teal)', fontFamily: 'var(--font-mono)' }}>
                              {domain2}
                            </span>
                          )}
                        </div>
                        <div style={{ position: 'relative', display: 'flex', alignItems: 'center' }}>
                          <input
                            type="url"
                            dir="ltr"
                            value={diffUrl2}
                            onChange={(e) => setDiffUrl2(cleanPastedUrl(e.target.value))}
                            placeholder={diffUrl1 ? isAr ? 'أدخل رابط الصفحة التالية للمقارنة (أو دعه فارغاً للاستنتاج الذكي)' : 'Enter next page URL' : 'https://arxiv.org/list/cs.AI/recent?skip=25&show=25'}
                            style={{
                              width: '100%',
                              padding: isAr ? '0.55rem 2rem 0.55rem 0.75rem' : '0.55rem 0.75rem 0.55rem 2rem',
                              background: 'var(--bg-secondary)',
                              border: hasHostMismatch ? '1px solid var(--accent-amber)' : '1px solid var(--border-medium)',
                              borderRadius: 'var(--radius-sm, 6px)',
                              color: 'var(--text-primary)',
                              fontFamily: 'var(--font-mono)',
                              fontSize: '0.82rem',
                              textAlign: 'left',
                              direction: 'ltr',
                            }}
                          />
                          {diffUrl2 && (
                            <button
                              type="button"
                              onClick={() => setDiffUrl2('')}
                              title={isAr ? 'مسح الرابط' : 'Clear URL'}
                              style={{
                                position: 'absolute',
                                [isAr ? 'right' : 'left']: '0.45rem',
                                background: 'transparent',
                                border: 'none',
                                color: 'var(--text-muted)',
                                cursor: 'pointer',
                                padding: '0.2rem',
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                              }}
                            >
                              <X size={14} />
                            </button>
                          )}
                        </div>
                      </div>
                    </div>

                    {/* Domain Mismatch Warning */}
                    {hasHostMismatch && (
                      <div
                        style={{
                          display: 'flex',
                          alignItems: 'center',
                          gap: '0.5rem',
                          background: 'rgba(245, 158, 11, 0.1)',
                          border: '1px solid rgba(245, 158, 11, 0.35)',
                          borderRadius: 'var(--radius-sm, 6px)',
                          padding: '0.55rem 0.85rem',
                          color: 'var(--accent-amber)',
                          fontSize: '0.8rem',
                        }}
                      >
                        <AlertTriangle size={15} style={{ flexShrink: 0 }} />
                        <span>
                          {isAr
                            ? `تنبيه: الرابط الأول يتبع لـ (${domain1}) بينما الرابط الثاني يتبع لـ (${domain2}). يجب أن يكون الرابطان لنفس الموقع لمقارنة رقم الصفحة بدقة.`
                            : `Warning: First URL is from (${domain1}) and second is from (${domain2}). Both URLs should be from the same website.`}
                        </span>
                      </div>
                    )}

                    {/* Action & Clear Toolbar */}
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.65rem', paddingTop: '0.25rem' }}>
                      <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
                        {(diffUrl1 || diffUrl2) && (
                          <button
                            type="button"
                            onClick={() => {
                              setDiffUrl1('');
                              setDiffUrl2('');
                              setInferError(null);
                              setInferredPattern(null);
                            }}
                            className="btn btn-ghost btn-sm"
                            style={{
                              fontSize: '0.76rem',
                              color: 'var(--text-muted)',
                              padding: '0.35rem 0.65rem',
                              display: 'inline-flex',
                              alignItems: 'center',
                              gap: '0.35rem',
                            }}
                          >
                            <Trash2 size={12} />
                            <span>{isAr ? 'تفريغ الحقول' : 'Clear Fields'}</span>
                          </button>
                        )}
                      </div>

                      <button
                        onClick={handleInferPattern}
                        disabled={inferLoading || !diffUrl1.trim()}
                        className="btn btn-primary"
                        style={{ padding: '0.55rem 1.5rem', fontWeight: 800, display: 'inline-flex', alignItems: 'center', gap: '0.45rem' }}
                      >
                        <Sparkles size={14} />
                        <span>{inferLoading ? (isAr ? 'جاري تحليل الرابط...' : 'Analyzing URL...') : (isAr ? 'تحليل واستنتاج النمط آلياً' : 'Infer Pattern Automatically')}</span>
                      </button>
                    </div>
                  </div>
                );
              })()}

              {/* Method 2: Auto DOM Discovery */}
              {patternMethod === 'auto' && (
                <div style={{ background: 'var(--bg-tertiary)', padding: '1rem 1.25rem', borderRadius: 'var(--radius-md, 8px)', border: '1px solid var(--border-subtle)', display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                  <div>
                    <label style={{ display: 'block', fontSize: '0.78rem', fontWeight: 700, color: 'var(--text-secondary)', marginBottom: '0.35rem' }}>
                      {isAr ? 'رابط صفحة البداية:' : 'Starting Page URL:'}
                    </label>
                    <input
                      type="url"
                      dir="ltr"
                      value={autoStartUrl}
                      onChange={(e) => setAutoStartUrl(cleanPastedUrl(e.target.value))}
                      placeholder="https://openlibrary.org/search?q=machine+learning"
                      style={{
                        width: '100%',
                        padding: '0.55rem 0.75rem',
                        background: 'var(--bg-secondary)',
                        border: '1px solid var(--border-medium)',
                        borderRadius: 'var(--radius-sm, 6px)',
                        color: 'var(--text-primary)',
                        fontFamily: 'var(--font-mono)',
                        fontSize: '0.82rem',
                        textAlign: 'left',
                        direction: 'ltr',
                      }}
                    />
                  </div>

                  <button
                    onClick={handleInferPattern}
                    disabled={inferLoading || !autoStartUrl.trim()}
                    className="btn btn-primary"
                    style={{ alignSelf: 'flex-start', padding: '0.55rem 1.4rem', fontWeight: 800, display: 'inline-flex', alignItems: 'center', gap: '0.45rem' }}
                  >
                    <Search size={14} />
                    <span>{inferLoading ? (isAr ? 'جاري فحص الصفحة...' : 'Inspecting Page...') : (isAr ? 'فحص الصفحة واكتشاف روابط الترقيم آلياً' : 'Auto-Discover Pagination')}</span>
                  </button>
                </div>
              )}

              {/* Method 3: Direct Template */}
              {patternMethod === 'template' && (
                <div style={{ background: 'var(--bg-tertiary)', padding: '1rem 1.25rem', borderRadius: 'var(--radius-md, 8px)', border: '1px solid var(--border-subtle)', display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                  <div>
                    <label style={{ display: 'block', fontSize: '0.78rem', fontWeight: 700, color: 'var(--text-secondary)', marginBottom: '0.35rem' }}>
                      {isAr ? 'قالب الرابط مع استبدال رقم الصفحة بـ {page}:' : 'URL Template with {page} Placeholder:'}
                    </label>
                    <input
                      type="text"
                      dir="ltr"
                      value={manualTemplate}
                      onChange={(e) => setManualTemplate(e.target.value)}
                      placeholder="https://arxiv.org/list/cs.AI/recent?skip={page}&show=25"
                      style={{
                        width: '100%',
                        padding: '0.55rem 0.75rem',
                        background: 'var(--bg-secondary)',
                        border: '1px solid var(--border-medium)',
                        borderRadius: 'var(--radius-sm, 6px)',
                        color: 'var(--text-primary)',
                        fontFamily: 'var(--font-mono)',
                        fontSize: '0.82rem',
                        textAlign: 'left',
                        direction: 'ltr',
                      }}
                    />
                  </div>

                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: '0.75rem' }}>
                    <div>
                      <label style={{ display: 'block', fontSize: '0.75rem', fontWeight: 700, color: 'var(--text-secondary)', marginBottom: '0.25rem' }}>
                        {isAr ? 'بداية الترقيم:' : 'Start Page:'}
                      </label>
                      <input
                        type="number"
                        min="0"
                        value={crawlStartPage}
                        onChange={(e) => setCrawlStartPage(e.target.value)}
                        style={{
                          width: '100%',
                          padding: '0.45rem 0.65rem',
                          background: 'var(--bg-secondary)',
                          border: '1px solid var(--border-medium)',
                          borderRadius: 'var(--radius-sm, 6px)',
                          color: 'var(--text-primary)',
                          fontSize: '0.85rem',
                        }}
                      />
                    </div>
                    <div>
                      <label style={{ display: 'block', fontSize: '0.75rem', fontWeight: 700, color: 'var(--text-secondary)', marginBottom: '0.25rem' }}>
                        {isAr ? 'مقدار الخطوة (Step):' : 'Step Increment:'}
                      </label>
                      <input
                        type="number"
                        min="1"
                        value={crawlStep}
                        onChange={(e) => setCrawlStep(e.target.value)}
                        style={{
                          width: '100%',
                          padding: '0.45rem 0.65rem',
                          background: 'var(--bg-secondary)',
                          border: '1px solid var(--border-medium)',
                          borderRadius: 'var(--radius-sm, 6px)',
                          color: 'var(--text-primary)',
                          fontSize: '0.85rem',
                        }}
                      />
                    </div>
                  </div>

                  <button
                    onClick={handleInferPattern}
                    disabled={inferLoading || !manualTemplate.trim()}
                    className="btn btn-primary"
                    style={{ alignSelf: 'flex-start', padding: '0.55rem 1.4rem', fontWeight: 800, display: 'inline-flex', alignItems: 'center', gap: '0.45rem' }}
                  >
                    <Check size={14} />
                    <span>{isAr ? 'اعتماد القالب وتوليد العينات' : 'Verify Template & Generate Samples'}</span>
                  </button>
                </div>
              )}

              {/* Error Message */}
              {inferError && (
                <div style={{ padding: '0.75rem 1rem', background: 'rgba(239, 68, 68, 0.1)', border: '1px solid var(--accent-rose, #ef4444)', borderRadius: 'var(--radius-sm, 6px)', color: 'var(--accent-rose, #ef4444)', fontSize: '0.82rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <AlertCircle size={15} />
                  <span>{inferError}</span>
                </div>
              )}

              {/* Inferred Pattern Verification & Execution Card */}
              {inferredPattern && (
                <div
                  style={{
                    background: 'var(--bg-tertiary)',
                    border: '1px solid var(--accent-teal)',
                    borderRadius: 'var(--radius-md, 8px)',
                    padding: '1.25rem',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '1rem',
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '0.5rem' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                      <CheckCircle2 size={16} color="var(--accent-teal)" />
                      <span style={{ fontWeight: 800, fontSize: '0.92rem', color: 'var(--text-primary)' }}>
                        {isAr ? 'تم استنتاج نمط الترقيم بنجاح' : 'URL Pattern Inferred Successfully'}
                      </span>
                      <span className="badge badge-teal" style={{ fontSize: '0.72rem', fontFamily: 'var(--font-mono)' }}>
                        {inferredPattern.detected_param || 'page'}
                      </span>
                    </div>
                    <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                      {isAr ? inferredPattern.explanation_ar : inferredPattern.explanation_en}
                    </span>
                  </div>

                  {/* Template Display */}
                  <div style={{ background: 'var(--bg-secondary)', padding: '0.65rem 0.85rem', borderRadius: 'var(--radius-sm, 6px)', border: '1px solid var(--border-medium)', fontFamily: 'var(--font-mono)', fontSize: '0.82rem', color: 'var(--accent-teal)', wordBreak: 'break-all', direction: 'ltr', textAlign: 'left' }}>
                    {inferredPattern.template}
                  </div>

                  {/* Sample URLs */}
                  {inferredPattern.sample_urls && inferredPattern.sample_urls.length > 0 && (
                    <div>
                      <span style={{ fontSize: '0.76rem', fontWeight: 700, color: 'var(--text-secondary)', display: 'block', marginBottom: '0.35rem' }}>
                        {isAr ? 'معاينة الروابط المتسلسلة الناتجة:' : 'Sample Generated Sequence:'}
                      </span>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
                        {inferredPattern.sample_urls.map((u, uIdx) => (
                          <div key={uIdx} style={{ fontSize: '0.76rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', direction: 'ltr', textAlign: 'left', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                            <span style={{ color: 'var(--accent-teal)', fontWeight: 800 }}>#{uIdx + 1}</span>
                            <span>{u}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Crawling Execution & Anti-Ban Controls */}
                  <div style={{ paddingTop: '0.75rem', borderTop: '1px solid var(--border-subtle)', display: 'flex', flexDirection: 'column', gap: '0.85rem' }}>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: '0.75rem', alignItems: 'flex-end' }}>
                      <div>
                        <label style={{ display: 'block', fontSize: '0.75rem', fontWeight: 700, color: 'var(--text-secondary)', marginBottom: '0.25rem' }}>
                          {isAr ? 'عدد الصفحات المطلوب سحبها:' : 'Number of Pages to Crawl:'}
                        </label>
                        <input
                          type="number"
                          min="1"
                          max="25"
                          value={crawlNumPages}
                          onChange={(e) => setCrawlNumPages(e.target.value)}
                          style={{
                            width: '100%',
                            padding: '0.45rem 0.65rem',
                            background: 'var(--bg-secondary)',
                            border: '1px solid var(--border-medium)',
                            borderRadius: 'var(--radius-sm, 6px)',
                            color: 'var(--text-primary)',
                            fontSize: '0.85rem',
                          }}
                        />
                      </div>

                      <div>
                        <label style={{ display: 'block', fontSize: '0.75rem', fontWeight: 700, color: 'var(--text-secondary)', marginBottom: '0.25rem' }}>
                          {isAr ? 'نوع استخراج المحتوى:' : 'Extraction Mode:'}
                        </label>
                        <select
                          value={crawlExtractMode}
                          onChange={(e) => setCrawlExtractMode(e.target.value)}
                          style={{
                            width: '100%',
                            padding: '0.45rem 0.65rem',
                            background: 'var(--bg-secondary)',
                            border: '1px solid var(--border-medium)',
                            borderRadius: 'var(--radius-sm, 6px)',
                            color: 'var(--text-primary)',
                            fontSize: '0.85rem',
                          }}
                        >
                          <option value="auto">{isAr ? 'كشف تلقائي شامل (موصى به: جداول أو بطاقات)' : 'Comprehensive Smart Auto (Recommended)'}</option>
                          <option value="cards">{isAr ? 'بطاقات ومشاريع ديناميكية (Cards / Items)' : 'Dynamic Cards / Project Items'}</option>
                          <option value="tables">{isAr ? 'جداول فقط (HTML Tables)' : 'Tables Only'}</option>
                          <option value="text">{isAr ? 'نصوص وفقرات (Text / Paragraphs)' : 'Text / Paragraphs'}</option>
                        </select>
                      </div>

                      <div>
                        <label style={{ display: 'block', fontSize: '0.75rem', fontWeight: 700, color: 'var(--text-secondary)', marginBottom: '0.25rem' }}>
                          {isAr ? 'الفاصل الزمني بين الصفحات (ث):' : 'Polite Delay (s):'}
                        </label>
                        <input
                          type="number"
                          step="0.5"
                          min="0.2"
                          max="10"
                          value={crawlDelay}
                          onChange={(e) => setCrawlDelay(e.target.value)}
                          style={{
                            width: '100%',
                            padding: '0.45rem 0.65rem',
                            background: 'var(--bg-secondary)',
                            border: '1px solid var(--border-medium)',
                            borderRadius: 'var(--radius-sm, 6px)',
                            color: 'var(--text-primary)',
                            fontSize: '0.85rem',
                          }}
                        />
                      </div>

                      <div>
                        <label style={{ display: 'block', fontSize: '0.75rem', fontWeight: 700, color: 'var(--text-secondary)', marginBottom: '0.25rem' }}>
                          {isAr ? 'انتظار تصيير العناصر (ث):' : 'Dynamic Render Wait (s):'}
                        </label>
                        <input
                          type="number"
                          step="0.5"
                          min="0.5"
                          max="8"
                          value={crawlWaitTime}
                          onChange={(e) => setCrawlWaitTime(e.target.value)}
                          style={{
                            width: '100%',
                            padding: '0.45rem 0.65rem',
                            background: 'var(--bg-secondary)',
                            border: '1px solid var(--border-medium)',
                            borderRadius: 'var(--radius-sm, 6px)',
                            color: 'var(--text-primary)',
                            fontSize: '0.85rem',
                          }}
                        />
                      </div>
                    </div>

                    {/* Anti-Ban and Stealth Options Toolbar */}
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.75rem', padding: '0.55rem 0.85rem', background: 'var(--bg-secondary)', borderRadius: 'var(--radius-sm, 6px)', border: '1px solid var(--border-subtle)', fontSize: '0.78rem' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '1.25rem', flexWrap: 'wrap' }}>
                        <label style={{ display: 'inline-flex', alignItems: 'center', gap: '0.35rem', cursor: 'pointer', color: 'var(--text-secondary)' }}>
                          <input
                            type="checkbox"
                            checked={crawlEnableJitter}
                            onChange={(e) => setCrawlEnableJitter(e.target.checked)}
                            style={{ accentColor: 'var(--accent-teal)' }}
                          />
                          <span>{isAr ? 'تفعيل العشوائية الذكية للفاصل الزمني (Anti-Ban Jitter)' : 'Enable Random Anti-Ban Jitter'}</span>
                        </label>

                        <label style={{ display: 'inline-flex', alignItems: 'center', gap: '0.35rem', cursor: 'pointer', color: 'var(--text-secondary)' }}>
                          <input
                            type="checkbox"
                            checked={crawlUseStealth}
                            onChange={(e) => setCrawlUseStealth(e.target.checked)}
                            style={{ accentColor: 'var(--accent-teal)' }}
                          />
                          <span>{isAr ? 'المتصفح الشبح الخفي (Headless Stealth)' : 'Stealth Headless Mode'}</span>
                        </label>
                      </div>

                      <button
                        onClick={handleExecuteMultiScrape}
                        disabled={multiScrapeLoading}
                        className="btn btn-primary"
                        style={{ padding: '0.55rem 1.6rem', fontWeight: 800, display: 'inline-flex', alignItems: 'center', justifyContent: 'center', gap: '0.45rem' }}
                      >
                        <Play size={15} />
                        <span>{multiScrapeLoading ? (isAr ? `جاري السحب... (${scrapeElapsedSeconds} ثانية)` : `Harvesting... (${scrapeElapsedSeconds}s)`) : (isAr ? 'بدء حلقة السحب المتعدد' : 'Start Multi-Page Crawl')}</span>
                      </button>
                    </div>

                    {/* Inline Multi-Page Error / Diagnostic Banner */}
                    {multiScrapeError && (
                      <div style={{ padding: '0.75rem 1rem', background: 'rgba(239, 68, 68, 0.12)', border: '1px solid var(--accent-rose, #ef4444)', borderRadius: 'var(--radius-sm, 6px)', color: 'var(--accent-rose, #ef4444)', fontSize: '0.82rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        <AlertCircle size={16} style={{ flexShrink: 0 }} />
                        <span>{multiScrapeError}</span>
                      </div>
                    )}
                  </div>

                  {/* Multi-Scrape Page-by-Page Progress Status */}
                  {multiScrapeStats && (
                    <div style={{ marginTop: '0.5rem', display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <span style={{ fontSize: '0.78rem', fontWeight: 700, color: 'var(--text-primary)' }}>
                          {isAr ? 'سجل المرور على الصفحات وتوقيت كل مرحلة:' : 'Crawled Pages Summary & Timing:'}
                        </span>
                        {scrapeTotalDuration && (
                          <span style={{ fontSize: '0.75rem', color: 'var(--accent-teal)', fontFamily: 'var(--font-mono)', fontWeight: 700 }}>
                            {isAr ? `إجمالي مدة السحب: ${scrapeTotalDuration} ثانية` : `Total Time: ${scrapeTotalDuration}s`}
                          </span>
                        )}
                      </div>
                      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.45rem' }}>
                        {multiScrapeStats.map((st) => (
                          <div
                            key={st.page_index}
                            style={{
                              padding: '0.35rem 0.75rem',
                              borderRadius: 'var(--radius-sm, 6px)',
                              background: st.status === 'success' ? 'rgba(20, 184, 166, 0.15)' : 'rgba(239, 68, 68, 0.15)',
                              border: `1px solid ${st.status === 'success' ? 'var(--accent-teal)' : 'var(--accent-rose, #ef4444)'}`,
                              fontSize: '0.74rem',
                              fontFamily: 'var(--font-mono)',
                              display: 'inline-flex',
                              alignItems: 'center',
                              gap: '0.5rem',
                            }}
                          >
                            <span style={{ fontWeight: 800 }}>#{st.page_index} (val={st.page_val})</span>
                            <span style={{ color: st.status === 'success' ? 'var(--accent-teal)' : 'var(--accent-rose, #ef4444)' }}>
                              {st.rows_found} {isAr ? 'صفاً' : 'rows'}
                            </span>
                            {st.duration_seconds !== undefined && (
                              <span style={{ opacity: 0.8, fontSize: '0.7rem' }}>⏱ {st.duration_seconds}s</span>
                            )}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          {/* Scraped Table Preview */}
          {scrapedPreview && (
            <div
              style={{
                background: 'var(--bg-tertiary)',
                border: '1px solid var(--accent-teal)',
                borderRadius: 'var(--radius-md, 8px)',
                overflow: 'auto',
                resize: 'both',
                minWidth: '320px',
                minHeight: '200px',
                marginTop: '1rem',
                boxShadow: 'var(--shadow-md)',
                transition: 'border-color 0.3s ease',
                position: 'relative',
              }}
            >
              {/* Header Bar Handle */}
              <div
                onClick={() => setScrapedPreviewOpen(!scrapedPreviewOpen)}
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  padding: '0.65rem 1.25rem',
                  background: 'var(--bg-secondary)',
                  cursor: 'pointer',
                  borderBottom: scrapedPreviewOpen ? '1px solid var(--border-medium)' : 'none',
                  userSelect: 'none',
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                  <span style={{ color: 'var(--accent-teal)', display: 'inline-flex', alignItems: 'center' }}>
                    {scrapedPreviewOpen ? (
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                        <polyline points="18 15 12 9 6 15"></polyline>
                      </svg>
                    ) : (
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                        <polyline points="6 9 12 15 18 9"></polyline>
                      </svg>
                    )}
                  </span>
                  <span style={{ fontWeight: 800, fontSize: '0.92rem', color: 'var(--text-primary)' }}>
                    {isAr ? 'معاينة الجدول المستخرج بنجاح' : 'Extracted Table Preview'}
                  </span>
                  <span className="badge badge-teal" style={{ fontSize: '0.72rem', fontFamily: 'var(--font-mono)' }}>
                    {scrapedPreview.rows.length} {isAr ? 'صفاً' : 'rows'} · {scrapedPreview.columns.length} {isAr ? 'أعمدة' : 'cols'}
                  </span>
                  {scrapeTotalDuration && (
                    <span className="badge badge-amber" style={{ fontSize: '0.72rem', fontFamily: 'var(--font-mono)' }}>
                      ⏱ {scrapeTotalDuration}s
                    </span>
                  )}
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: '0.85rem' }} onClick={(e) => e.stopPropagation()}>
                  <button
                    onClick={handleImportScraped}
                    className="btn btn-primary btn-sm"
                    style={{ fontWeight: 800, padding: '0.35rem 1rem', fontSize: '0.78rem', display: 'inline-flex', alignItems: 'center', gap: '0.4rem' }}
                  >
                    <span>{isAr ? 'استيراد هذا الجدول إلى الاستوديو' : 'Import Table'}</span>
                    <ArrowRight size={13} style={{ transform: isAr ? 'scaleX(-1)' : 'none' }} />
                  </button>
                  <button
                    onClick={() => setScrapedPreviewOpen(!scrapedPreviewOpen)}
                    className="btn btn-ghost btn-sm"
                    style={{ color: 'var(--accent-teal)', fontSize: '0.78rem', fontWeight: 800, padding: '0.2rem 0.5rem', display: 'inline-flex', alignItems: 'center', gap: '0.3rem' }}
                  >
                    <span>{scrapedPreviewOpen ? (isAr ? 'طي المعاينة' : 'Collapse') : (isAr ? 'توسيع المعاينة' : 'Expand')}</span>
                    {scrapedPreviewOpen ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
                  </button>
                </div>
              </div>

              {/* Table Body & Search */}
              {scrapedPreviewOpen && (() => {
                const filteredRows = scrapedPreview.rows.filter(r => {
                  if (!scrapedSearch.trim()) return true;
                  const query = scrapedSearch.toLowerCase();
                  return scrapedPreview.columns.some(col => String(r[col] || '').toLowerCase().includes(query));
                });
                const pageSize = 5;
                const totalPages = Math.max(1, Math.ceil(filteredRows.length / pageSize));
                const currentPageSlice = filteredRows.slice((scrapedPage - 1) * pageSize, scrapedPage * pageSize);

                return (
                  <div style={{ padding: '0.85rem 1.25rem 0.65rem 1.25rem', background: 'var(--bg-tertiary)' }}>
                    {/* Search bar inside preview */}
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.65rem' }}>
                      <input
                        type="text"
                        placeholder={isAr ? 'بحث فوري في خلايا الجدول...' : 'Search preview cells...'}
                        value={scrapedSearch}
                        onChange={(e) => {
                          setScrapedSearch(e.target.value);
                          setScrapedPage(1);
                        }}
                        style={{
                          padding: '0.3rem 0.75rem',
                          borderRadius: 'var(--radius-sm, 6px)',
                          border: '1px solid var(--border-medium)',
                          background: 'var(--bg-secondary)',
                          color: 'var(--text-primary)',
                          fontSize: '0.76rem',
                          width: '240px',
                        }}
                      />
                      <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
                        {isAr ? `إجمالي المطابقات: ${filteredRows.length} صفاً` : `Matches: ${filteredRows.length} rows`}
                      </span>
                    </div>

                    <div style={{ overflowX: 'auto', maxHeight: '300px', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-sm, 6px)' }}>
                      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.78rem' }}>
                        <thead>
                          <tr style={{ background: 'var(--bg-secondary)', borderBottom: '1px solid var(--border-medium)', position: 'sticky', top: 0, zIndex: 10 }}>
                            <th style={{ padding: '0.55rem 0.75rem', color: 'var(--text-muted)', width: '45px', textAlign: 'center' }}>#</th>
                            {scrapedPreview.columns.map((col) => (
                              <th key={col} style={{ padding: '0.55rem 0.75rem', textAlign: 'start', color: 'var(--accent-teal)', fontFamily: 'var(--font-mono)', fontWeight: 800 }}>
                                {col}
                              </th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {currentPageSlice.map((r, idx) => (
                            <tr
                              key={idx}
                              style={{
                                borderBottom: '1px solid var(--border-subtle)',
                                background: idx % 2 === 0 ? 'var(--bg-secondary)' : 'var(--bg-tertiary)',
                              }}
                            >
                              <td style={{ padding: '0.45rem 0.65rem', color: 'var(--text-muted)', textAlign: 'center', fontFamily: 'var(--font-mono)' }}>
                                {(scrapedPage - 1) * pageSize + idx + 1}
                              </td>
                              {scrapedPreview.columns.map((col) => {
                                const valStr = String(r[col] || '');
                                return (
                                  <td
                                    key={col}
                                    title={valStr}
                                    style={{
                                      padding: '0.45rem 0.65rem',
                                      color: 'var(--text-primary)',
                                      maxWidth: '320px',
                                      overflow: 'hidden',
                                      textOverflow: 'ellipsis',
                                      whiteSpace: 'nowrap',
                                    }}
                                  >
                                    {valStr || '—'}
                                  </td>
                                );
                              })}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>

                    {/* Pagination Bar */}
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '0.65rem', fontSize: '0.75rem' }}>
                      <span style={{ color: 'var(--text-secondary)' }}>
                        {isAr ? `الصفحة ${scrapedPage} من ${totalPages}` : `Page ${scrapedPage} of ${totalPages}`}
                      </span>
                      <div style={{ display: 'flex', gap: '0.4rem' }}>
                        <button
                          onClick={() => setScrapedPage(Math.max(1, scrapedPage - 1))}
                          disabled={scrapedPage === 1}
                          className="btn btn-secondary btn-sm"
                          style={{ padding: '0.2rem 0.65rem', fontSize: '0.75rem' }}
                        >
                          {isAr ? 'السابق' : 'Prev'}
                        </button>
                        <button
                          onClick={() => setScrapedPage(Math.min(totalPages, scrapedPage + 1))}
                          disabled={scrapedPage >= totalPages}
                          className="btn btn-secondary btn-sm"
                          style={{ padding: '0.2rem 0.65rem', fontSize: '0.75rem' }}
                        >
                          {isAr ? 'التالي' : 'Next'}
                        </button>
                      </div>
                    </div>
                  </div>
                );
              })()}
            </div>
          )}
        </div>
      )}

      {/* Source 3: Global Open Datasets Search */}
      {activeSource === 'search' && (
        <div
          style={{
            background: 'var(--bg-secondary)',
            border: '1px solid var(--border-subtle)',
            borderRadius: 'var(--radius-lg, 12px)',
            padding: '1.5rem',
            display: 'flex',
            flexDirection: 'column',
            gap: '1.25rem',
          }}
        >
          <div>
            <h3 style={{ fontSize: '1.1rem', fontWeight: 800, color: 'var(--text-primary)', marginBottom: '0.35rem' }}>
              {isAr ? 'البحث في قواعد البيانات العالمية المفتوحة (Global Datasets Search)' : 'Search Global Public Dataset Repositories'}
            </h3>
            <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', margin: 0 }}>
              {isAr
                ? 'ابحث في أشهر المستودعات العالمية (UCI, Data.gov, World Bank, GitHub, Kaggle) واستورد قواعد بيانات حقيقية مباشرة بضغطة زر.'
                : 'Query top global repositories (UCI, Data.gov, World Bank, GitHub, Kaggle) and import real datasets.'}
            </p>
          </div>

          {/* Platform Filtering Chips with Guidance Tooltips (Multi-Select Support & Zero Emojis) */}
          <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap', alignItems: 'center' }}>
            <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)', fontWeight: 700 }}>
              {isAr ? 'تصفية حسب المنصة (يمكن تحديد أكثر من منصة):' : 'Filter Platforms (Multi-Select):'}
            </span>

            {[
              { id: 'all', termKey: 'all_repos', label_ar: 'الكل', label_en: 'All Repositories' },
              { id: 'google', termKey: 'platform_google', label_ar: 'المستودعات المفتوحة', label_en: 'Open Repositories' },
              { id: 'huggingface', termKey: 'huggingface', label_ar: 'Hugging Face', label_en: 'Hugging Face' },
              { id: 'kaggle', termKey: 'kaggle', label_ar: 'Kaggle Datasets', label_en: 'Kaggle Datasets' },
              { id: 'github', termKey: 'github', label_ar: 'Awesome / GitHub', label_en: 'Awesome / GitHub' },
              { id: 'uci', termKey: 'uci', label_ar: 'UCI ML Repo', label_en: 'UCI ML Repo' },
              { id: 'datagov', termKey: 'datagov', label_ar: 'Data.gov', label_en: 'Data.gov' },
              { id: 'worldbank', termKey: 'worldbank', label_ar: 'World Bank', label_en: 'World Bank' },
            ].map((p) => {
              const isSelected = searchPlatformFilters.includes('all') ? p.id === 'all' : searchPlatformFilters.includes(p.id);
              return (
                <div key={p.id} style={{ display: 'inline-flex', alignItems: 'center', gap: '0.35rem' }}>
                  <button
                    type="button"
                    onClick={() => handleTogglePlatformFilter(p.id)}
                    style={{
                      padding: '0.35rem 0.8rem',
                      borderRadius: 'var(--radius-sm, 6px)',
                      border: isSelected ? '1px solid var(--accent-teal)' : '1px solid var(--border-subtle)',
                      background: isSelected ? 'var(--accent-teal-glow)' : 'var(--bg-tertiary)',
                      color: isSelected ? 'var(--accent-teal)' : 'var(--text-secondary)',
                      fontSize: '0.78rem',
                      fontWeight: isSelected ? 800 : 600,
                      cursor: 'pointer',
                      transition: 'all 0.15s ease',
                    }}
                  >
                    {isAr ? p.label_ar : p.label_en}
                  </button>
                  <TermTooltip termKey={p.termKey} hideLabel={true} size="sm" />
                </div>
              );
            })}
          </div>

          <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap', alignItems: 'center' }}>
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              onKeyDown={(e) => { if (e.key === 'Enter' && searchQuery.trim()) handleSearch(); }}
              placeholder={isAr ? 'ابحث عن موضوع أو داتا (مثال: رسائل مزعجة, مشاعر عربية, iris, diabetes, gdp)...' : 'Search query (e.g. spam sms, arabic sentiment, iris, diabetes, housing)...'}
              style={{
                flex: 1,
                minWidth: '260px',
                padding: '0.65rem 1rem',
                background: 'var(--bg-tertiary)',
                border: '1px solid var(--border-medium)',
                borderRadius: 'var(--radius-sm, 6px)',
                color: 'var(--text-primary)',
                fontSize: '0.85rem',
              }}
            />

            {/* User Input: Number of Datasets (userinput_n) */}
            <div
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '0.45rem',
                background: 'var(--bg-tertiary)',
                border: '1px solid var(--border-medium)',
                borderRadius: 'var(--radius-sm, 6px)',
                padding: '0.35rem 0.75rem',
                flexShrink: 0,
              }}
              title={isAr ? 'حدد عدد قواعد البيانات المراد استرجاعها من المستودعات (1 - 50)' : 'Number of datasets to retrieve (1 - 50)'}
            >
              <Database size={15} style={{ color: 'var(--accent-teal)', flexShrink: 0 }} />
              <label style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', fontWeight: 700, margin: 0, whiteSpace: 'nowrap' }}>
                {isAr ? 'العدد الأقصى:' : 'Max Datasets:'}
              </label>
              <input
                type="number"
                min="1"
                max="50"
                value={searchLimit}
                onChange={(e) => {
                  const val = e.target.value;
                  if (val === '') {
                    setSearchLimit('');
                  } else {
                    const parsed = parseInt(val, 10);
                    if (!isNaN(parsed)) {
                      setSearchLimit(Math.min(50, Math.max(1, parsed)));
                    }
                  }
                }}
                onBlur={() => {
                  if (!searchLimit || searchLimit < 1) setSearchLimit(20);
                }}
                onKeyDown={(e) => { if (e.key === 'Enter' && searchQuery.trim()) handleSearch(); }}
                style={{
                  width: '46px',
                  padding: '0.28rem 0.2rem',
                  background: 'var(--bg-secondary)',
                  border: '1px solid var(--border-subtle)',
                  borderRadius: 'var(--radius-xs, 4px)',
                  color: 'var(--accent-teal)',
                  fontSize: '0.85rem',
                  fontWeight: 800,
                  fontFamily: 'var(--font-mono)',
                  textAlign: 'center',
                  outline: 'none',
                }}
              />
            </div>

            <button
              onClick={handleSearch}
              disabled={searchLoading || !searchQuery.trim()}
              className="btn btn-primary"
              style={{ padding: '0.65rem 1.75rem', fontWeight: 800, whiteSpace: 'nowrap' }}
            >
              {searchLoading ? (isAr ? 'جاري البحث في المستودعات...' : 'Searching Repos...') : (isAr ? 'بحث في المستودعات' : 'Search Datasets')}
            </button>
          </div>

          {/* Search Notice */}
          {searchError && (
            <div
              style={{
                padding: '0.75rem 1.15rem',
                background: searchResults.length > 0 ? 'rgba(245, 158, 11, 0.08)' : 'rgba(239, 68, 68, 0.08)',
                border: searchResults.length > 0 ? '1px solid rgba(245, 158, 11, 0.35)' : '1px solid var(--accent-rose)',
                borderRadius: 'var(--radius-md, 8px)',
                color: searchResults.length > 0 ? 'var(--text-secondary)' : 'var(--accent-rose)',
                fontSize: '0.82rem',
                display: 'flex',
                alignItems: 'center',
                gap: '0.65rem',
                lineHeight: 1.5,
              }}
            >
              <AlertCircle size={18} style={{ flexShrink: 0, color: searchResults.length > 0 ? '#f59e0b' : 'var(--accent-rose)' }} />
              <div style={{ flex: 1 }}>
                <span style={{ fontWeight: 800, color: searchResults.length > 0 ? '#f59e0b' : 'var(--accent-rose)' }}>
                  {isAr ? 'تنبيه الاتصال بالمستودعات:' : 'Repository Query Notice:'}
                </span>{' '}
                <span>{searchError}</span>
              </div>
            </div>
          )}

          {searchSuccessMsg && (
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '0.65rem',
                padding: '0.75rem 1rem',
                background: 'rgba(16, 185, 129, 0.12)',
                border: '1px solid var(--accent-teal, #10b981)',
                borderRadius: 'var(--radius-sm, 6px)',
                fontSize: '0.84rem',
                color: 'var(--text-primary)',
              }}
            >
              <Check size={18} style={{ color: 'var(--accent-teal, #10b981)', flexShrink: 0 }} />
              <div style={{ flex: 1, fontWeight: 700 }}>
                {searchSuccessMsg}
              </div>
            </div>
          )}

          {/* Pre-Fetch Sample Preview Modal Box */}
          {previewingDataset && (
            <div
              style={{
                background: 'var(--bg-tertiary)',
                border: '1px solid var(--accent-teal)',
                borderRadius: 'var(--radius-md, 8px)',
                padding: '1.25rem',
                boxShadow: 'var(--shadow-md)',
                overflow: 'auto',
                resize: 'both',
                minWidth: '320px',
                minHeight: '220px',
                position: 'relative',
                boxSizing: 'border-box',
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem', flexWrap: 'wrap', gap: '0.5rem' }}>
                <div style={{ minWidth: 0, maxWidth: 'calc(100% - 130px)' }}>
                  <span className="badge badge-teal" style={{ marginBottom: '0.25rem', fontSize: '0.7rem' }}>
                    {isAr ? 'معاينة عينة البيانات قبل الاعتماد النهائي' : 'Light Metadata & Sample Preview'}
                  </span>
                  <h4
                    title={previewingDataset.title}
                    style={{
                      margin: 0,
                      fontSize: '1rem',
                      fontWeight: 800,
                      color: 'var(--text-primary)',
                      overflowWrap: 'anywhere',
                      wordBreak: 'break-word',
                    }}
                  >
                    {previewingDataset.title}
                  </h4>
                  <span style={{ fontSize: '0.76rem', color: 'var(--text-muted)' }}>
                    {isAr ? `المنصة: ${previewingDataset.platform_name}` : `Source: ${previewingDataset.platform_name}`}
                  </span>
                </div>

                <button
                  onClick={() => {
                    setPreviewingDataset(null);
                    setRemotePreviewResult(null);
                  }}
                  className="btn btn-secondary btn-sm"
                  style={{ fontSize: '0.75rem', display: 'inline-flex', alignItems: 'center', gap: '0.35rem', flexShrink: 0 }}
                >
                  <span>{isAr ? 'إلغاء المعاينة' : 'Close Preview'}</span>
                  <X size={13} />
                </button>
              </div>

              {fetchingRemote ? (
                <div style={{ padding: '1.5rem', textAlign: 'center', color: 'var(--accent-teal)', fontWeight: 700, fontSize: '0.88rem' }}>
                  {isAr ? 'جاري جلب عينة خفيفة من البيانات دون استهلاك الحزمة...' : 'Fetching light dataset sample...'}
                </div>
              ) : (() => {
                const isGarbageHtml =
                  remotePreviewResult &&
                  remotePreviewResult.rows &&
                  remotePreviewResult.rows.some((r) => {
                    const val = String(Object.values(r || {})[0] || '').toLowerCase();
                    return val.includes('<!doctype') || val.includes('<html') || val.includes('<script') || val.includes('window.google');
                  });

                const isSearchPortal =
                  remotePreviewResult &&
                  (remotePreviewResult.status === 'search_portal' ||
                    remotePreviewResult.is_search_portal ||
                    (previewingDataset?.platform === 'google' && !previewingDataset?.download_url && isGarbageHtml));

                const isRealSuccess =
                  remotePreviewResult &&
                  remotePreviewResult.status === 'success' &&
                  !isGarbageHtml &&
                  !isSearchPortal;

                if (isRealSuccess) {
                  return (
                    <div>
                      <div style={{ display: 'flex', gap: '1rem', marginBottom: '0.75rem', fontSize: '0.8rem', color: 'var(--text-secondary)', flexWrap: 'wrap' }}>
                        <span><strong>{isAr ? 'عدد الصفوف المتاحة:' : 'Rows:'}</strong> {remotePreviewResult.rows_count}</span>
                        <span><strong>{isAr ? 'عدد الأعمدة:' : 'Columns:'}</strong> {remotePreviewResult.cols_count}</span>
                        <span><strong>{isAr ? 'اسم الملف:' : 'Filename:'}</strong> <code style={{ color: 'var(--accent-teal)', wordBreak: 'break-all' }}>{remotePreviewResult.filename}</code></span>
                      </div>

                      <div style={{ overflowX: 'auto', maxHeight: '180px', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-sm, 4px)', marginBottom: '1rem' }}>
                        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.76rem', textAlign: 'start' }}>
                          <thead>
                            <tr style={{ background: 'var(--bg-secondary)', borderBottom: '1px solid var(--border-medium)', position: 'sticky', top: 0 }}>
                              {remotePreviewResult.columns.map((col) => (
                                <th key={col} style={{ padding: '0.45rem 0.65rem', color: 'var(--accent-teal)', fontFamily: 'var(--font-mono)', fontWeight: 800, whiteSpace: 'nowrap' }}>
                                  {col}
                                </th>
                              ))}
                            </tr>
                          </thead>
                          <tbody>
                            {remotePreviewResult.rows.slice(0, 5).map((r, idx) => (
                              <tr key={idx} style={{ borderBottom: '1px solid var(--border-subtle)', background: idx % 2 === 0 ? 'var(--bg-tertiary)' : 'var(--bg-secondary)' }}>
                                {remotePreviewResult.columns.map((col) => (
                                  <td key={col} style={{ padding: '0.4rem 0.65rem', color: 'var(--text-primary)', whiteSpace: 'nowrap', maxWidth: '200px', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                                    {String(r[col] ?? '')}
                                  </td>
                                ))}
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>

                      <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '0.75rem', flexWrap: 'wrap' }}>
                        <button
                          onClick={handleConfirmRemoteImport}
                          className="btn btn-primary"
                          style={{ fontWeight: 800, padding: '0.55rem 1.5rem', fontSize: '0.85rem', display: 'inline-flex', alignItems: 'center', gap: '0.45rem' }}
                        >
                          <span>{isAr ? 'تأكيد التحميل والاستيراد إلى الاستوديو' : 'Confirm Download & Load Dataset'}</span>
                          <ArrowRight size={15} />
                        </button>
                      </div>
                    </div>
                  );
                }

                if (remotePreviewResult) {
                  return (
                    <div
                      style={{
                        padding: '1.15rem',
                        background: 'var(--bg-secondary)',
                        border: '1px solid var(--border-medium)',
                        borderRadius: 'var(--radius-sm, 6px)',
                        display: 'flex',
                        flexDirection: 'column',
                        gap: '0.85rem',
                      }}
                    >
                      <div style={{ display: 'flex', alignItems: 'flex-start', gap: '0.65rem' }}>
                        <AlertCircle size={20} style={{ color: isSearchPortal ? 'var(--accent-teal)' : 'var(--accent-rose)', flexShrink: 0, marginTop: '2px' }} />
                        <div style={{ minWidth: 0 }}>
                          <h5 style={{ margin: '0 0 0.35rem 0', color: 'var(--text-primary)', fontSize: '0.9rem', fontWeight: 800 }}>
                            {isSearchPortal
                              ? (isAr ? 'بوابة استعلام وبحث خارجي (Search Portal)' : 'External Search Portal Notice')
                              : (isAr ? 'تنبيه الاتصال بالمستودع الخارجي' : 'External Repository Notice')}
                          </h5>
                          <p style={{ margin: 0, fontSize: '0.82rem', color: 'var(--text-secondary)', lineHeight: 1.5, wordBreak: 'break-word' }}>
                            {isSearchPortal
                              ? (isAr
                                ? 'هذا الرابط يمثل بوابة بحث واستعلام على الويب وليس ملف بيانات CSV مباشر. يرجى النقر على الزر أدناه لفتح صفحة البحث في علامة تبويب جديدة لاختيار الملف المناسب وتنزيله، ثم استيراده مباشرة إلى الاستوديو.'
                                : 'This link points to an external search index. Please open the repository page to download your target CSV and import it into the Studio.')
                              : remotePreviewResult.message}
                          </p>
                        </div>
                      </div>

                      <div style={{ display: 'flex', gap: '0.65rem', flexWrap: 'wrap', justifyContent: 'flex-end', paddingTop: '0.5rem', borderTop: '1px solid var(--border-subtle)' }}>
                        <button
                          onClick={handleInstantSynthesizeSample}
                          className="btn btn-primary btn-sm"
                          style={{
                            fontSize: '0.8rem',
                            fontWeight: 800,
                            display: 'inline-flex',
                            alignItems: 'center',
                            gap: '0.4rem',
                            background: 'var(--accent-teal, #10b981)',
                            borderColor: 'var(--accent-teal, #10b981)',
                            color: '#fff',
                          }}
                        >
                          <Zap size={14} />
                          <span>{isAr ? 'توليد واستيراد عينة أكاديمية فورية داخل المنصة' : 'Synthesize In-Platform Academic Sample'}</span>
                        </button>
                        {previewingDataset?.external_url && (
                          <button
                            onClick={() => window.open(previewingDataset.external_url, '_blank')}
                            className="btn btn-secondary btn-sm"
                            style={{ fontSize: '0.8rem', fontWeight: 800, display: 'inline-flex', alignItems: 'center', gap: '0.4rem' }}
                          >
                            <span>{isAr ? 'الانتقال إلى صفحة المستودع الأصلية ↗' : 'Open Repository Page ↗'}</span>
                            <ExternalLink size={13} />
                          </button>
                        )}
                        <button
                          onClick={() => {
                            setPreviewingDataset(null);
                            setRemotePreviewResult(null);
                            setActiveSource('upload');
                          }}
                          className="btn btn-ghost btn-sm"
                          style={{ fontSize: '0.8rem', fontWeight: 700 }}
                        >
                          {isAr ? 'رفع ملف محلي بدلاً من ذلك' : 'Upload Local File Instead'}
                        </button>
                      </div>
                    </div>
                  );
                }

                return null;
              })()}
            </div>
          )}

          {/* Search Results Display */}
          {searchResults.length > 0 && (() => {
            const filteredResults = searchResults.filter((r) => {
              if (searchPlatformFilters.includes('all')) return true;
              return searchPlatformFilters.includes(r.platform);
            });

            const totalSearchPages = Math.max(1, Math.ceil(filteredResults.length / SEARCH_PAGE_SIZE));
            const validSearchPage = Math.min(searchPage, totalSearchPages);
            const currentSearchSlice = filteredResults.slice(
              (validSearchPage - 1) * SEARCH_PAGE_SIZE,
              validSearchPage * SEARCH_PAGE_SIZE
            );

            return (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem', width: '100%', maxWidth: '100%', boxSizing: 'border-box' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.5rem' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <Database size={16} style={{ color: 'var(--accent-teal)' }} />
                    <span style={{ fontSize: '0.88rem', fontWeight: 800, color: 'var(--text-primary)' }}>
                      {isAr ? `قواعد البيانات المرصودة (${filteredResults.length})` : `Discovered Datasets (${filteredResults.length})`}
                    </span>
                  </div>

                  {filteredResults.length > SEARCH_PAGE_SIZE && (
                    <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                      {isAr
                        ? `عرض ${currentSearchSlice.length} من أصل ${filteredResults.length} قاعدة بيانات`
                        : `Showing ${currentSearchSlice.length} of ${filteredResults.length} datasets`}
                    </span>
                  )}
                </div>

                {/* 4 Items Per Page Grid with Absolute Zero Overflow */}
                <div
                  style={{
                    display: 'grid',
                    gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
                    gap: '0.85rem',
                    width: '100%',
                    maxWidth: '100%',
                    boxSizing: 'border-box',
                  }}
                >
                  {currentSearchSlice.map((res) => (
                    <div
                      key={res.result_id}
                      style={{
                        padding: '1.15rem',
                        background: 'var(--bg-tertiary)',
                        borderRadius: 'var(--radius-md, 8px)',
                        border: '1px solid var(--border-subtle)',
                        display: 'flex',
                        flexDirection: 'column',
                        justifyContent: 'space-between',
                        gap: '0.75rem',
                        boxShadow: 'var(--shadow-sm)',
                        minWidth: 0,
                        maxWidth: '100%',
                        overflow: 'hidden',
                        boxSizing: 'border-box',
                      }}
                    >
                      <div style={{ minWidth: 0, overflow: 'hidden' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: '0.5rem', marginBottom: '0.45rem', flexWrap: 'wrap' }}>
                          <span
                            className="badge badge-neutral"
                            style={{
                              fontSize: '0.68rem',
                              fontFamily: 'var(--font-mono)',
                              maxWidth: 'calc(100% - 90px)',
                              overflow: 'hidden',
                              textOverflow: 'ellipsis',
                              whiteSpace: 'nowrap',
                            }}
                          >
                            {res.platform_name}
                          </span>
                          {res.is_direct_downloadable ? (
                            <span className="badge badge-teal" style={{ fontSize: '0.68rem', fontWeight: 800, flexShrink: 0 }}>
                              {isAr ? 'تحميل مباشر جاهز' : 'Direct CSV'}
                            </span>
                          ) : (
                            <span className="badge badge-neutral" style={{ fontSize: '0.68rem', flexShrink: 0 }}>
                              {res.format || 'Repository'}
                            </span>
                          )}
                        </div>

                        <h4
                          title={res.title}
                          style={{
                            margin: '0 0 0.35rem 0',
                            fontSize: '0.92rem',
                            fontWeight: 800,
                            color: 'var(--text-primary)',
                            lineHeight: 1.4,
                            overflowWrap: 'anywhere',
                            wordBreak: 'break-word',
                            display: '-webkit-box',
                            WebkitLineClamp: 2,
                            WebkitBoxOrient: 'vertical',
                            overflow: 'hidden',
                          }}
                        >
                          {res.title}
                        </h4>

                        <p
                          title={res.description}
                          style={{
                            margin: 0,
                            fontSize: '0.8rem',
                            color: 'var(--text-secondary)',
                            lineHeight: 1.5,
                            overflowWrap: 'anywhere',
                            wordBreak: 'break-word',
                            display: '-webkit-box',
                            WebkitLineClamp: 3,
                            WebkitBoxOrient: 'vertical',
                            overflow: 'hidden',
                          }}
                        >
                          {res.description}
                        </p>
                      </div>

                      <div
                        style={{
                          paddingTop: '0.65rem',
                          borderTop: '1px solid var(--border-subtle)',
                          display: 'flex',
                          justifyContent: 'space-between',
                          alignItems: 'center',
                          flexWrap: 'wrap',
                          gap: '0.5rem',
                        }}
                      >
                        <span style={{ fontSize: '0.74rem', color: 'var(--text-muted)', flexShrink: 0 }}>
                          ~{Number(res.estimated_rows || 1000).toLocaleString()} {isAr ? 'صفاً متوقعاً' : 'rows'}
                        </span>

                        <div style={{ display: 'flex', gap: '0.45rem', alignItems: 'center', flexWrap: 'wrap' }}>
                          <button
                            onClick={() => handleFetchSamplePreview(res)}
                            className="btn btn-primary btn-sm"
                            style={{ fontWeight: 800, padding: '0.4rem 0.95rem', fontSize: '0.8rem', display: 'inline-flex', alignItems: 'center', gap: '0.35rem', flexShrink: 0 }}
                            title={isAr ? 'معاينة عينة من البيانات وتأكيد استيرادها المباشر إلى المنصة' : 'Preview sample and import directly into DYTR Studio'}
                          >
                            <span>{isAr ? 'معاينة واستيراد البيانات' : 'Preview & Import'}</span>
                            <ArrowRight size={13} />
                          </button>

                          {res.external_url && (
                            <button
                              onClick={() => window.open(res.external_url, '_blank')}
                              className="btn btn-secondary btn-sm"
                              style={{ fontSize: '0.75rem', fontWeight: 700, padding: '0.35rem 0.65rem', display: 'inline-flex', alignItems: 'center', gap: '0.25rem', opacity: 0.85 }}
                              title={isAr ? 'فتح صفحة المستودع الأصلية في علامة تبويب خارجية' : 'Open original repository page in external tab'}
                            >
                              <span>{isAr ? 'المصدر ↗' : 'Source ↗'}</span>
                              <ExternalLink size={11} />
                            </button>
                          )}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>

                {/* Pagination Controls Bar (4 Datasets Per Page) */}
                {totalSearchPages > 1 && (
                  <div
                    style={{
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'center',
                      padding: '0.75rem 1rem',
                      background: 'var(--bg-tertiary)',
                      borderRadius: 'var(--radius-sm, 6px)',
                      border: '1px solid var(--border-subtle)',
                      marginTop: '0.5rem',
                      fontSize: '0.8rem',
                      flexWrap: 'wrap',
                      gap: '0.65rem',
                      boxSizing: 'border-box',
                    }}
                  >
                    <span style={{ color: 'var(--text-secondary)', fontWeight: 600 }}>
                      {isAr
                        ? `صفحة النتائج ${validSearchPage} من ${totalSearchPages}`
                        : `Results Page ${validSearchPage} of ${totalSearchPages}`}
                    </span>

                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
                      <button
                        onClick={() => setSearchPage(Math.max(1, validSearchPage - 1))}
                        disabled={validSearchPage <= 1}
                        className="btn btn-secondary btn-sm"
                        style={{
                          padding: '0.28rem 0.75rem',
                          fontSize: '0.78rem',
                          display: 'inline-flex',
                          alignItems: 'center',
                          gap: '0.3rem',
                          opacity: validSearchPage <= 1 ? 0.45 : 1,
                          cursor: validSearchPage <= 1 ? 'not-allowed' : 'pointer',
                        }}
                      >
                        <ChevronRight size={14} style={{ transform: isAr ? 'none' : 'rotate(180deg)' }} />
                        <span>{isAr ? 'السابق' : 'Prev'}</span>
                      </button>

                      {/* Numeric Page Buttons */}
                      {Array.from({ length: totalSearchPages }, (_, i) => i + 1).map((pNum) => (
                        <button
                          key={pNum}
                          onClick={() => setSearchPage(pNum)}
                          style={{
                            minWidth: '28px',
                            height: '28px',
                            padding: '0 0.35rem',
                            borderRadius: 'var(--radius-sm, 4px)',
                            border: pNum === validSearchPage ? '1px solid var(--accent-teal)' : '1px solid var(--border-subtle)',
                            background: pNum === validSearchPage ? 'var(--accent-teal-glow)' : 'transparent',
                            color: pNum === validSearchPage ? 'var(--accent-teal)' : 'var(--text-secondary)',
                            fontWeight: pNum === validSearchPage ? 800 : 600,
                            fontSize: '0.78rem',
                            cursor: 'pointer',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            fontFamily: 'var(--font-mono)',
                          }}
                        >
                          {pNum}
                        </button>
                      ))}

                      <button
                        onClick={() => setSearchPage(Math.min(totalSearchPages, validSearchPage + 1))}
                        disabled={validSearchPage >= totalSearchPages}
                        className="btn btn-secondary btn-sm"
                        style={{
                          padding: '0.28rem 0.75rem',
                          fontSize: '0.78rem',
                          display: 'inline-flex',
                          alignItems: 'center',
                          gap: '0.3rem',
                          opacity: validSearchPage >= totalSearchPages ? 0.45 : 1,
                          cursor: validSearchPage >= totalSearchPages ? 'not-allowed' : 'pointer',
                        }}
                      >
                        <span>{isAr ? 'التالي' : 'Next'}</span>
                        <ChevronLeft size={14} style={{ transform: isAr ? 'none' : 'rotate(180deg)' }} />
                      </button>
                    </div>
                  </div>
                )}
              </div>
            );
          })()}
        </div>
      )}

      {/* Source 4: Synthetic Data Generation */}
      {activeSource === 'synthetic' && (
        <div
          style={{
            background: 'var(--bg-secondary)',
            border: '1px solid var(--border-subtle)',
            borderRadius: 'var(--radius-lg, 12px)',
            padding: '1.5rem',
            display: 'flex',
            flexDirection: 'column',
            gap: '1.25rem',
          }}
        >
          <div>
            <h3 style={{ fontSize: '1.1rem', fontWeight: 800, color: 'var(--text-primary)', marginBottom: '0.35rem' }}>
              <TermTooltip termKey="synthetic_data" customLabel={isAr ? 'محرك توليد البيانات الاصطناعية الذكية والإدخال المباشر' : 'Parametric Synthetic Generator & Manual Data Matrix'} />
            </h3>
            <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', margin: 0 }}>
              {isAr
                ? 'توليد نصوص عربية وبيانات إحصائية واقعية متوازنة، أو بناء أعمدة مخصصة وفق شروطك، أو لصق البيانات يدوياً.'
                : 'Generate balanced Arabic NLP corpora, construct custom constraint-guided features, or input matrices manually.'}
            </p>
          </div>

          {/* Sub-Tabs: Presets vs Guided Custom Builder vs Manual Entry */}
          <div style={{ display: 'flex', gap: '0.5rem', borderBottom: '1px solid var(--border-subtle)', paddingBottom: '0.5rem', overflowX: 'auto' }}>
            {[
              { id: 'presets', label_ar: 'قوالب أكاديمية جاهزة', label_en: 'Academic Presets' },
              { id: 'guided', label_ar: 'توليد موجه بأعمدة مخصصة (Guided Builder)', label_en: 'Custom Guided Builder' },
              { id: 'manual', label_ar: 'محرر الإدخال واللصق اليدوي (Manual Matrix)', label_en: 'Manual Matrix Editor' },
            ].map((st) => (
              <button
                key={st.id}
                onClick={() => setSyntheticTab(st.id)}
                style={{
                  padding: '0.45rem 1rem',
                  borderRadius: 'var(--radius-sm, 6px)',
                  border: 'none',
                  background: syntheticTab === st.id ? 'var(--accent-teal)' : 'var(--bg-tertiary)',
                  color: syntheticTab === st.id ? '#ffffff' : 'var(--text-secondary)',
                  fontWeight: syntheticTab === st.id ? '800' : '600',
                  fontSize: '0.82rem',
                  cursor: 'pointer',
                  whiteSpace: 'nowrap',
                }}
              >
                {isAr ? st.label_ar : st.label_en}
              </button>
            ))}
          </div>

          {/* TAB 4.1: Quick Presets */}
          {syntheticTab === 'presets' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '0.85rem' }}>
                {[
                  { id: 'sentiment', title_ar: 'تحليل المشاعر العربية', desc_ar: 'نصوص مراجعات مع 3 فئات ومقاييس تقييم' },
                  { id: 'news', title_ar: 'تصنيف الأخبار العربية', desc_ar: 'مقالات موزعة على 4 مجالات معرفية' },
                  { id: 'titanic', title_ar: 'سجلات ديموغرافية (Titanic)', desc_ar: 'بيانات رقمية وفئوية مع قيم مفقودة تجريبية' },
                  { id: 'iris', title_ar: 'أزهار فيشر (Fisher Iris)', desc_ar: '150 عينة كلاسيكية للتعلم الآلي' },
                ].map((item) => (
                  <div
                    key={item.id}
                    onClick={() => setSyntheticType(item.id)}
                    style={{
                      padding: '1rem',
                      borderRadius: 'var(--radius-md, 8px)',
                      border: syntheticType === item.id ? '1px solid var(--accent-teal)' : '1px solid var(--border-subtle)',
                      background: syntheticType === item.id ? 'var(--accent-teal-glow)' : 'var(--bg-tertiary)',
                      cursor: 'pointer',
                    }}
                  >
                    <div style={{ fontWeight: 800, fontSize: '0.9rem', color: 'var(--text-primary)', marginBottom: '0.25rem' }}>
                      {item.title_ar}
                    </div>
                    <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
                      {item.desc_ar}
                    </div>
                  </div>
                ))}
              </div>

              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                    {isAr ? 'عدد الصفوف المطلوبة:' : 'Sample Size:'}
                  </span>
                  <select
                    value={syntheticCount}
                    onChange={(e) => setSyntheticCount(Number(e.target.value))}
                    style={{
                      padding: '0.45rem 0.75rem',
                      background: 'var(--bg-tertiary)',
                      border: '1px solid var(--border-medium)',
                      borderRadius: 'var(--radius-sm, 6px)',
                      color: 'var(--text-primary)',
                      fontSize: '0.85rem',
                    }}
                  >
                    <option value={30}>30 {isAr ? 'سجل' : 'rows'}</option>
                    <option value={60}>60 {isAr ? 'سجل' : 'rows'}</option>
                    <option value={120}>120 {isAr ? 'سجل' : 'rows'}</option>
                    <option value={200}>200 {isAr ? 'سجل' : 'rows'}</option>
                  </select>
                </div>

                <button
                  onClick={handleGeneratePreset}
                  className="btn btn-primary"
                  style={{ padding: '0.65rem 1.75rem', fontWeight: 800, display: 'inline-flex', alignItems: 'center', gap: '0.5rem' }}
                >
                  <span>{isAr ? 'توليد واستيراد البيانات القالبية' : 'Generate & Load Preset'}</span>
                  <ArrowRight size={15} style={{ transform: isAr ? 'scaleX(-1)' : 'none' }} />
                </button>
              </div>
            </div>
          )}

          {/* TAB 4.2: Custom Guided Builder */}
          {syntheticTab === 'guided' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '1rem' }}>
                <div>
                  <label style={{ display: 'block', fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '0.35rem' }}>
                    {isAr ? 'اسم مجموعة البيانات' : 'Dataset Name'}
                  </label>
                  <input
                    type="text"
                    value={customDatasetName}
                    onChange={(e) => setCustomDatasetName(e.target.value)}
                    style={{
                      width: '100%',
                      padding: '0.5rem 0.75rem',
                      background: 'var(--bg-tertiary)',
                      border: '1px solid var(--border-medium)',
                      borderRadius: 'var(--radius-sm, 6px)',
                      color: 'var(--text-primary)',
                      fontSize: '0.85rem',
                    }}
                  />
                </div>

                <div>
                  <label style={{ display: 'block', fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '0.35rem' }}>
                    {isAr ? 'عدد العينات المراد توليدها' : 'Number of Rows'}
                  </label>
                  <input
                    type="number"
                    min={5}
                    max={500}
                    value={customRowsCount}
                    onChange={(e) => setCustomRowsCount(Number(e.target.value))}
                    style={{
                      width: '100%',
                      padding: '0.5rem 0.75rem',
                      background: 'var(--bg-tertiary)',
                      border: '1px solid var(--border-medium)',
                      borderRadius: 'var(--radius-sm, 6px)',
                      color: 'var(--text-primary)',
                      fontSize: '0.85rem',
                    }}
                  />
                </div>
              </div>

              {/* Column Definitions List */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                <span style={{ fontSize: '0.85rem', fontWeight: 800, color: 'var(--text-primary)' }}>
                  {isAr ? 'تحديد بنية الأعمدة المخصصة:' : 'Column Topology Configuration:'}
                </span>

                {customColumns.map((col, idx) => (
                  <div
                    key={col.id}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '0.75rem',
                      padding: '0.75rem',
                      background: 'var(--bg-tertiary)',
                      borderRadius: 'var(--radius-md, 8px)',
                      border: '1px solid var(--border-subtle)',
                      flexWrap: 'wrap',
                    }}
                  >
                    <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)', width: '20px', fontFamily: 'var(--font-mono)' }}>
                      #{idx + 1}
                    </span>

                    <input
                      type="text"
                      placeholder={isAr ? 'اسم العمود' : 'Column Name'}
                      value={col.name}
                      onChange={(e) => handleUpdateCustomCol(col.id, 'name', e.target.value)}
                      style={{
                        padding: '0.45rem 0.75rem',
                        borderRadius: 'var(--radius-sm, 6px)',
                        border: '1px solid var(--border-medium)',
                        background: 'var(--bg-secondary)',
                        color: 'var(--text-primary)',
                        fontSize: '0.85rem',
                        minWidth: '130px',
                      }}
                    />

                    <select
                      value={col.type}
                      onChange={(e) => handleUpdateCustomCol(col.id, 'type', e.target.value)}
                      style={{
                        padding: '0.45rem 0.75rem',
                        borderRadius: 'var(--radius-sm, 6px)',
                        border: '1px solid var(--border-medium)',
                        background: 'var(--bg-secondary)',
                        color: 'var(--text-primary)',
                        fontSize: '0.85rem',
                      }}
                    >
                      <option value="arabic_text">{isAr ? 'نصوص عربية (Arabic Text)' : 'Arabic Text'}</option>
                      <option value="category">{isAr ? 'تصنيف فئوي (Category)' : 'Category Factor'}</option>
                      <option value="numeric">{isAr ? 'قيمة رقمية (Numeric Continuous)' : 'Numeric Continuous'}</option>
                      <option value="date">{isAr ? 'تاريخ زمني (Date)' : 'Date (ISO)'}</option>
                    </select>

                    {col.type === 'category' && (
                      <input
                        type="text"
                        placeholder={isAr ? 'الفئات مفصولة بفواصل (e.g. نعم, لا)' : 'Categories (comma separated)'}
                        value={col.categories || ''}
                        onChange={(e) => handleUpdateCustomCol(col.id, 'categories', e.target.value)}
                        style={{
                          flex: 1,
                          minWidth: '160px',
                          padding: '0.45rem 0.75rem',
                          borderRadius: 'var(--radius-sm, 6px)',
                          border: '1px solid var(--border-medium)',
                          background: 'var(--bg-secondary)',
                          color: 'var(--text-primary)',
                          fontSize: '0.85rem',
                        }}
                      />
                    )}

                    {col.type === 'numeric' && (
                      <div style={{ display: 'flex', gap: '0.35rem', alignItems: 'center' }}>
                        <input
                          type="number"
                          placeholder="Min"
                          value={col.min !== undefined ? col.min : 0}
                          onChange={(e) => handleUpdateCustomCol(col.id, 'min', e.target.value)}
                          style={{
                            width: '70px',
                            padding: '0.45rem',
                            borderRadius: 'var(--radius-sm, 6px)',
                            border: '1px solid var(--border-medium)',
                            background: 'var(--bg-secondary)',
                            color: 'var(--text-primary)',
                            fontSize: '0.85rem',
                          }}
                        />
                        <span style={{ color: 'var(--text-muted)' }}>-</span>
                        <input
                          type="number"
                          placeholder="Max"
                          value={col.max !== undefined ? col.max : 100}
                          onChange={(e) => handleUpdateCustomCol(col.id, 'max', e.target.value)}
                          style={{
                            width: '70px',
                            padding: '0.45rem',
                            borderRadius: 'var(--radius-sm, 6px)',
                            border: '1px solid var(--border-medium)',
                            background: 'var(--bg-secondary)',
                            color: 'var(--text-primary)',
                            fontSize: '0.85rem',
                          }}
                        />
                      </div>
                    )}

                    {customColumns.length > 1 && (
                      <button
                        onClick={() => handleRemoveCustomCol(col.id)}
                        className="btn btn-danger btn-sm"
                        style={{ padding: '0.4rem 0.65rem' }}
                      >
                        {isAr ? 'حذف' : 'Drop'}
                      </button>
                    )}
                  </div>
                ))}
              </div>

              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '0.5rem', flexWrap: 'wrap', gap: '0.5rem' }}>
                <button
                  onClick={handleAddCustomCol}
                  className="btn btn-secondary btn-sm"
                  style={{ color: 'var(--accent-teal)', borderStyle: 'dashed' }}
                >
                  {isAr ? '+ إضافة عمود جديد' : '+ Add Column Dimension'}
                </button>

                <button
                  onClick={handleGenerateCustomGuided}
                  className="btn btn-primary"
                  style={{ padding: '0.65rem 1.75rem', fontWeight: 800, display: 'inline-flex', alignItems: 'center', gap: '0.5rem' }}
                >
                  <span>{isAr ? 'توليد البيانات المخصصة واستيرادها' : 'Generate & Import Dataset'}</span>
                  <ArrowRight size={15} style={{ transform: isAr ? 'scaleX(-1)' : 'none' }} />
                </button>
              </div>
            </div>
          )}

          {/* TAB 4.3: Manual Matrix Editor */}
          {syntheticTab === 'manual' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
              <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', margin: 0 }}>
                {isAr
                  ? 'اكتب أو الصق بيانات CSV مباشرة في المحرر أدناه (مع سطر الترويسة):'
                  : 'Paste or write direct raw CSV data into the matrix editor:'}
              </p>

              <textarea
                rows={7}
                value={manualText}
                onChange={(e) => setManualText(e.target.value)}
                style={{
                  width: '100%',
                  padding: '1rem',
                  borderRadius: 'var(--radius-md, 8px)',
                  background: 'var(--bg-tertiary)',
                  color: 'var(--text-primary)',
                  fontFamily: 'var(--font-mono)',
                  fontSize: '0.82rem',
                  lineHeight: 1.5,
                  border: '1px solid var(--border-medium)',
                  direction: 'ltr',
                  textAlign: 'left',
                }}
              />

              <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
                <button
                  onClick={handleImportManual}
                  className="btn btn-primary"
                  style={{ padding: '0.65rem 1.75rem', fontWeight: 800, display: 'inline-flex', alignItems: 'center', gap: '0.5rem' }}
                >
                  <span>{isAr ? 'تحليل واستيراد النص كمجموعة بيانات' : 'Parse & Import Matrix'}</span>
                  <ArrowRight size={15} style={{ transform: isAr ? 'scaleX(-1)' : 'none' }} />
                </button>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Source 5: Classic Curated Academic Benchmarks */}
      {activeSource === 'samples' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1rem' }}>
          {sampleDatasetsCatalog.map((item) => (
            <div
              key={item.id}
              style={{
                background: 'var(--bg-secondary)',
                border: '1px solid var(--border-subtle)',
                borderRadius: 'var(--radius-md, 10px)',
                padding: '1.25rem',
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'space-between',
                gap: '1rem',
              }}
            >
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
                  <span className="badge badge-teal">{item.category.toUpperCase()}</span>
                  <span style={{ fontSize: '0.75rem', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>
                    {item.rows} rows · {item.cols} cols
                  </span>
                </div>
                <h4 style={{ fontSize: '0.98rem', fontWeight: 800, color: 'var(--text-primary)', marginBottom: '0.35rem' }}>
                  {isAr ? item.name_ar : item.name_en}
                </h4>
                <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', lineHeight: 1.5, margin: 0 }}>
                  {isAr ? item.description_ar : item.description_en}
                </p>
              </div>

              <div style={{ display: 'flex', gap: '0.5rem' }}>
                <button
                  onClick={() => {
                    let data = [];
                    if (item.id === 'arabic_sentiment') data = generateArabicSentiment(60);
                    else if (item.id === 'arabic_news_classification') data = generateArabicNews(50);
                    else if (item.id === 'iris_flowers') data = generateIrisData();
                    else if (item.id === 'titanic_survival') data = generateTitanicData(70);
                    else data = generateArabicSentiment(50);
                    setDataset(data, `${item.id}.csv`, 'sample');
                  }}
                  className="btn btn-primary btn-sm"
                  style={{ flex: 1, fontWeight: 800 }}
                >
                  {isAr ? 'تحميل واستيراد' : 'Load Dataset'}
                </button>

                <button
                  onClick={() => handleDownloadSampleCSV(item)}
                  className="btn btn-secondary btn-sm"
                  title={isAr ? 'تنزيل الملف كـ CSV' : 'Download CSV file'}
                  style={{ padding: '0.4rem 0.75rem', fontFamily: 'var(--font-mono)', fontSize: '0.8rem' }}
                >
                  {isAr ? 'تنزيل CSV' : 'Download CSV'}
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Source 6: Studio Saved Datasets Library Catalog */}
      {activeSource === 'library' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
          {/* Search & Counter Toolbar */}
          <div
            style={{
              background: 'var(--bg-secondary)',
              border: '1px solid var(--border-subtle)',
              borderRadius: 'var(--radius-md, 8px)',
              padding: '1rem 1.25rem',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              flexWrap: 'wrap',
              gap: '1rem',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem', flex: '1 1 300px' }}>
              <div style={{ position: 'relative', width: '100%', maxWidth: '380px' }}>
                <input
                  type="text"
                  value={librarySearch}
                  onChange={(e) => setLibrarySearch(e.target.value)}
                  placeholder={isAr ? 'بحث بالاسم أو الوسوم أو الوصف...' : 'Search by name, tags, or description...'}
                  style={{
                    width: '100%',
                    padding: '0.55rem 0.85rem',
                    paddingInlineStart: '2.2rem',
                    borderRadius: '6px',
                    border: '1px solid var(--border-medium)',
                    background: 'var(--bg-tertiary)',
                    color: 'var(--text-primary)',
                    fontSize: '0.85rem',
                  }}
                />
                <div style={{ position: 'absolute', top: '50%', transform: 'translateY(-50%)', insetInlineStart: '0.75rem', color: 'var(--text-muted)', display: 'flex', alignItems: 'center' }}>
                  <Search size={15} />
                </div>
              </div>
              {librarySearch && (
                <button
                  type="button"
                  onClick={() => setLibrarySearch('')}
                  className="btn btn-secondary btn-sm"
                  style={{ fontSize: '0.75rem' }}
                >
                  {isAr ? 'مسح' : 'Clear'}
                </button>
              )}
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <span className="badge badge-teal" style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem' }}>
                {filteredLibrary.length} {isAr ? 'مجموعة مطابقة' : 'matching datasets'}
              </span>
              <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
                ({isAr ? 'من إجمالي' : 'out of'} {savedLibrary?.length || 0})
              </span>
            </div>
          </div>

          {/* Cards Grid or Empty State */}
          {filteredLibrary && filteredLibrary.length > 0 ? (
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: '1rem' }}>
              {filteredLibrary.map((item) => (
                <div
                  key={item.id}
                  style={{
                    padding: '1.2rem',
                    borderRadius: 'var(--radius-md, 10px)',
                    border: '1px solid var(--border-subtle)',
                    background: 'var(--bg-secondary)',
                    display: 'flex',
                    flexDirection: 'column',
                    justifyContent: 'space-between',
                    gap: '1rem',
                    boxShadow: 'var(--shadow-sm)',
                    overflow: 'hidden',
                    position: 'relative',
                  }}
                >
                  <div style={{ minWidth: 0 }}>
                    {/* Header: Title and Date Badge */}
                    <div
                      style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'flex-start',
                        gap: '0.6rem',
                        marginBottom: '0.6rem',
                        flexWrap: 'wrap',
                      }}
                    >
                      <h4
                        style={{
                          margin: 0,
                          fontSize: '0.96rem',
                          fontWeight: 800,
                          color: 'var(--text-primary)',
                          wordBreak: 'break-word',
                          overflowWrap: 'anywhere',
                          flex: '1 1 170px',
                          minWidth: 0,
                          lineHeight: 1.4,
                        }}
                        dir="auto"
                      >
                        {item.name}
                      </h4>
                      <span
                        className="badge badge-neutral"
                        style={{
                          fontSize: '0.68rem',
                          fontFamily: 'var(--font-mono)',
                          whiteSpace: 'nowrap',
                          flexShrink: 0,
                        }}
                      >
                        {formatLibraryDate(item.savedAt || item.loadedAt)}
                      </span>
                    </div>

                    <p
                      style={{
                        margin: '0 0 0.65rem 0',
                        fontSize: '0.82rem',
                        color: 'var(--text-secondary)',
                        lineHeight: 1.5,
                        wordBreak: 'break-word',
                        overflowWrap: 'anywhere',
                      }}
                    >
                      {item.description || (isAr ? 'مجموعة بيانات نظيفة ومعالجة جاهزة للتدريب في DYTR Studio.' : 'Preprocessed dataset saved in studio library.')}
                    </p>

                    {item.tags && item.tags.length > 0 && (
                      <div style={{ display: 'flex', gap: '0.35rem', flexWrap: 'wrap', marginBottom: '0.75rem' }}>
                        {item.tags.map((tag) => (
                          <span
                            key={tag}
                            style={{
                              fontSize: '0.7rem',
                              padding: '0.15rem 0.45rem',
                              borderRadius: '4px',
                              background: 'var(--bg-tertiary)',
                              color: 'var(--accent-teal, #14b8a6)',
                              border: '1px solid var(--border-subtle)',
                              fontWeight: 600,
                              whiteSpace: 'nowrap',
                            }}
                          >
                            #{tag}
                          </span>
                        ))}
                      </div>
                    )}

                    <div
                      style={{
                        fontSize: '0.76rem',
                        color: 'var(--text-muted)',
                        fontFamily: 'var(--font-mono)',
                        display: 'flex',
                        alignItems: 'center',
                        flexWrap: 'wrap',
                        gap: '0.5rem',
                      }}
                    >
                      <span>{item.rowsCount || item.data?.length || 0} {isAr ? 'صفاً' : 'rows'}</span>
                      <span>•</span>
                      <span>{item.colsCount || (item.data && item.data[0] ? Object.keys(item.data[0]).length : 0)} {isAr ? 'أعمدة' : 'cols'}</span>
                      {item.sourceType && (
                        <>
                          <span>•</span>
                          <span className="badge badge-neutral" style={{ fontSize: '0.65rem' }}>{item.sourceType}</span>
                        </>
                      )}
                    </div>
                  </div>

                  <div style={{ display: 'flex', gap: '0.5rem', paddingTop: '0.5rem', borderTop: '1px solid var(--border-subtle)' }}>
                    <button
                      type="button"
                      onClick={() => setDataset(item.data, item.name, 'library', { id: item.id })}
                      className="btn btn-primary btn-sm"
                      style={{
                        flex: 1,
                        fontWeight: 800,
                        display: 'inline-flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        gap: '0.4rem',
                      }}
                    >
                      <Check size={14} />
                      <span>{isAr ? 'تحميل في المنصة' : 'Load Dataset'}</span>
                    </button>
                    <button
                      type="button"
                      onClick={() => deleteFromLibrary(item.id)}
                      className="btn btn-danger btn-sm"
                      style={{
                        padding: '0.4rem 0.75rem',
                        display: 'inline-flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        gap: '0.3rem',
                      }}
                      title={isAr ? 'حذف من المكتبة' : 'Delete from library'}
                    >
                      <Trash2 size={13} />
                      <span>{isAr ? 'حذف' : 'Delete'}</span>
                    </button>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div
              style={{
                padding: '3rem 2rem',
                textAlign: 'center',
                background: 'var(--bg-secondary)',
                border: '1px dashed var(--border-medium)',
                borderRadius: 'var(--radius-lg, 12px)',
              }}
            >
              <div style={{ width: '48px', height: '48px', borderRadius: '50%', background: 'var(--bg-tertiary)', display: 'inline-flex', alignItems: 'center', justifyContent: 'center', marginBottom: '1rem', color: 'var(--text-muted)' }}>
                <Bookmark size={22} />
              </div>
              <h3 style={{ fontSize: '1.05rem', fontWeight: 800, color: 'var(--text-primary)', marginBottom: '0.4rem' }}>
                {librarySearch
                  ? (isAr ? 'لا توجد مجموعات تطابق كلمة البحث' : 'No matching datasets found')
                  : (isAr ? 'مكتبة الاستوديو فارغة حالياً' : 'Studio Library is currently empty')}
              </h3>
              <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', maxWidth: '520px', margin: '0 auto', lineHeight: 1.5 }}>
                {librarySearch
                  ? (isAr ? 'جرب البحث بكلمات أخرى أو مسح مرشح البحث.' : 'Try different keywords or clear the search filter.')
                  : (isAr
                    ? 'يمكنك حفظ أي مجموعة بيانات معالجة ومفلترة مستقبلاً من مرحلة "التصدير والحفظ (05)" لتظهر هنا دائماً للتحميل بضغطة زر.'
                    : 'You can save any processed dataset from Phase 5 (Export) to make it readily available here anytime with 1-click.')}
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
