// ==============================================================================
// DyTrStudio Data Preparation: Phase 4 - Transformation & Relational Fusion
// Advanced Multi-Condition Filtering, Column Engineering, Joins, GroupBy
// Zero Emojis Policy - High Contrast Dual Theme - Pedagogical Guidance
// ==============================================================================

import React, { useState, useMemo } from 'react';
import { useDatasetStore } from '../../store/useDatasetStore';
import { useStudioStore } from '../../store/studioStore';
import TermTooltip from '../common/TermTooltip';
import OptionTipBox from '../common/OptionTipBox';
import {
  Check,
  X,
  Plus,
  Trash2,
  Filter,
  Columns,
  GitMerge,
  Layers,
  ArrowUpDown,
  Play,
  ArrowRight,
  ArrowLeft,
} from 'lucide-react';
import { filterRows, joinDatasets, groupByAggregate, executeRuleAction } from '../../lib/data/processor';
import { sampleDatasetsCatalog, getSampleDatasetById } from '../../lib/data/synthetic';

export default function TransformationPanel() {
  const { language } = useStudioStore();
  const isAr = language === 'ar';

  const {
    currentDataset,
    applyOperation,
    savedLibrary,
  } = useDatasetStore();

  const [activeTab, setActiveTab] = useState('filter'); // 'filter' | 'columns' | 'join' | 'groupby' | 'sort'

  // --- Rule & Action Engine State ---
  const [filterConditions, setFilterConditions] = useState([
    { id: 1, column: Object.keys(currentDataset[0] || {})[0] || '', op: '==', value: '' },
  ]);
  const [filterLogic, setFilterLogic] = useState('AND');
  const [ruleActionType, setRuleActionType] = useState('keep'); // 'keep' | 'drop' | 'set_value' | 'flag' | 'replace_text'
  const [ruleTargetCol, setRuleTargetCol] = useState(Object.keys(currentDataset[0] || {})[0] || '');
  const [ruleSetValue, setRuleSetValue] = useState('');
  const [ruleFlagCol, setRuleFlagCol] = useState('is_matched');
  const [ruleFlagVal, setRuleFlagVal] = useState('1');
  const [ruleElseVal, setRuleElseVal] = useState('0');
  const [ruleFindText, setRuleFindText] = useState('');
  const [ruleReplaceText, setRuleReplaceText] = useState('');

  // --- Feature Engineering State ---
  const [newColName, setNewColName] = useState('');
  const [feType, setFeType] = useState('math'); // 'math' | 'text_len' | 'word_count' | 'text_concat'
  const [mathColA, setMathColA] = useState(Object.keys(currentDataset[0] || {})[0] || '');
  const [mathOp, setMathOp] = useState('+');
  const [mathColB, setMathColB] = useState(Object.keys(currentDataset[0] || {})[1] || '');
  const [mathScalar, setMathScalar] = useState(1);
  const [mathMode, setMathMode] = useState('two_cols'); // 'two_cols' | 'scalar'

  // --- Relational Join State ---
  const [joinSource, setJoinSource] = useState('sample'); // 'sample' | 'library'
  const [selectedSampleId, setSelectedSampleId] = useState('titanic_survival');
  const [selectedLibId, setSelectedLibId] = useState('');
  const [joinType, setJoinType] = useState('inner'); // 'inner' | 'left' | 'right' | 'outer'
  const [leftKey, setLeftKey] = useState(Object.keys(currentDataset[0] || {})[0] || '');
  const [rightKey, setRightKey] = useState('');

  // --- GroupBy State ---
  const [groupCol, setGroupCol] = useState(Object.keys(currentDataset[0] || {})[0] || '');
  const [aggCol, setAggCol] = useState(Object.keys(currentDataset[0] || {})[1] || '');
  const [aggFn, setAggFn] = useState('mean'); // 'mean' | 'sum' | 'min' | 'max' | 'std'

  // --- Sort State ---
  const [sortCol, setSortCol] = useState(Object.keys(currentDataset[0] || {})[0] || '');
  const [sortOrder, setSortOrder] = useState('asc');
  const [actionToast, setActionToast] = useState(null);

  const triggerToast = (msg_ar, msg_en) => {
    setActionToast(isAr ? msg_ar : msg_en);
    setTimeout(() => setActionToast(null), 4500);
  };

  const columns = useMemo(
    () => Object.keys(currentDataset[0] || {}).filter((k) => !k.endsWith('_original')),
    [currentDataset]
  );

  // Secondary Dataset for Join
  const secondaryDataset = useMemo(() => {
    if (joinSource === 'sample') {
      return getSampleDatasetById(selectedSampleId);
    }
    const lib = savedLibrary.find((s) => s.id === selectedLibId);
    return lib ? lib.data : [];
  }, [joinSource, selectedSampleId, selectedLibId, savedLibrary]);

  const secondaryColumns = useMemo(() => Object.keys(secondaryDataset[0] || {}), [secondaryDataset]);

  // Sync selected columns when currentDataset columns change
  React.useEffect(() => {
    if (columns.length > 0) {
      if (!mathColA || !columns.includes(mathColA)) {
        setMathColA(columns[0]);
      }
      if (!mathColB || !columns.includes(mathColB)) {
        setMathColB(columns[1] || columns[0]);
      }
      if (!groupCol || !columns.includes(groupCol)) {
        setGroupCol(columns[0]);
      }
      if (!aggCol || !columns.includes(aggCol)) {
        setAggCol(columns[1] || columns[0]);
      }
      if (!sortCol || !columns.includes(sortCol)) {
        setSortCol(columns[0]);
      }
      if (!leftKey || !columns.includes(leftKey)) {
        setLeftKey(columns[0]);
      }
    }
  }, [columns]);

  // Sync rightKey when secondaryColumns change
  React.useEffect(() => {
    if (secondaryColumns.length > 0 && (!rightKey || !secondaryColumns.includes(rightKey))) {
      setRightKey(secondaryColumns[0]);
    }
  }, [secondaryColumns, rightKey]);

  // Sync selectedLibId when savedLibrary changes
  React.useEffect(() => {
    if (savedLibrary.length > 0 && !selectedLibId) {
      setSelectedLibId(savedLibrary[0].id);
    }
  }, [savedLibrary, selectedLibId]);

  // --- Handlers ---
  const handleAddCondition = () => {
    setFilterConditions([
      ...filterConditions,
      { id: Date.now(), column: columns[0] || '', op: '==', value: '' },
    ]);
  };

  const handleRemoveCondition = (id) => {
    if (filterConditions.length <= 1) return;
    setFilterConditions(filterConditions.filter((c) => c.id !== id));
  };

  const handleUpdateCondition = (id, field, val) => {
    setFilterConditions(
      filterConditions.map((c) => (c.id === id ? { ...c, [field]: val } : c))
    );
  };

  const handleApplyFilter = () => {
    const validConds = filterConditions.filter(
      (c) => c.column && (c.op === 'is_null' || c.op === 'not_null' || c.value !== '')
    );
    if (validConds.length === 0) return;

    const actionSpec = {
      actionType: ruleActionType,
      targetColumn: ruleTargetCol || columns[0],
      setValue: ruleSetValue,
      flagColumn: ruleFlagCol || 'is_matched',
      flagValue: ruleFlagVal,
      elseValue: ruleElseVal,
      findText: ruleFindText,
      replaceText: ruleReplaceText,
    };

    let titleAr = '';
    let titleEn = '';

    if (ruleActionType === 'keep') {
      titleAr = `إبقاء الصفوف المطابقة لـ (${validConds.map((c) => `${c.column} ${c.op} ${c.value}`).join(` ${filterLogic} `)})`;
      titleEn = `Filtered & kept rows matching (${validConds.map((c) => `${c.column} ${c.op} ${c.value}`).join(` ${filterLogic} `)})`;
    } else if (ruleActionType === 'drop') {
      titleAr = `حذف الصفوف المطابقة لـ (${validConds.map((c) => `${c.column} ${c.op} ${c.value}`).join(` ${filterLogic} `)})`;
      titleEn = `Dropped rows matching (${validConds.map((c) => `${c.column} ${c.op} ${c.value}`).join(` ${filterLogic} `)})`;
    } else if (ruleActionType === 'set_value') {
      titleAr = `تحديث [${ruleTargetCol}] إلى "${ruleSetValue}" للصفوف المطابقة للشروط`;
      titleEn = `Set column [${ruleTargetCol}] = "${ruleSetValue}" for matching rows`;
    } else if (ruleActionType === 'flag') {
      titleAr = `وسم الصفوف المطابقة في عمود [${ruleFlagCol}] بقيمة "${ruleFlagVal}"`;
      titleEn = `Flagged matching rows in column [${ruleFlagCol}]`;
    } else if (ruleActionType === 'replace_text') {
      titleAr = `استبدال "${ruleFindText}" بـ "${ruleReplaceText}" في [${ruleTargetCol}] للصفوف المطابقة`;
      titleEn = `Replaced text in [${ruleTargetCol}] for matching rows`;
    }

    applyOperation({
      type: 'rule_action',
      title_ar: titleAr,
      title_en: titleEn,
      conditions: validConds,
      filterLogic,
      actionSpec,
      transformFn: (data) => executeRuleAction(data, validConds, filterLogic, actionSpec),
    });

    triggerToast(
      'تم تنفيذ الإجراء الشرطي بنجاح وتحديث بيانات الصفوف.',
      'Conditional rule action applied successfully to dataset.'
    );
  };

  const handleAddFeature = () => {
    if (!newColName.trim() || !mathColA) return;

    const targetName = newColName.trim();

    if (feType === 'math') {
      applyOperation({
        type: 'feature_engineer',
        feType: 'math',
        column: targetName,
        mathColA,
        mathColB: mathMode === 'two_cols' ? mathColB : undefined,
        mathOp,
        mathMode,
        mathScalar: mathMode === 'scalar' ? mathScalar : undefined,
        title_ar: `إضافة عمود حسابي جديد: [${targetName}]`,
        title_en: `Created engineered feature: [${targetName}]`,
        transformFn: (data) =>
          data.map((row) => {
            const vA = Number(row[mathColA]) || 0;
            let res = 0;
            if (mathMode === 'two_cols') {
              const vB = Number(row[mathColB]) || 0;
              if (mathOp === '+') res = vA + vB;
              else if (mathOp === '-') res = vA - vB;
              else if (mathOp === '*') res = vA * vB;
              else if (mathOp === '/') res = vB !== 0 ? vA / vB : 0;
            } else {
              const s = Number(mathScalar) || 1;
              if (mathOp === '+') res = vA + s;
              else if (mathOp === '-') res = vA - s;
              else if (mathOp === '*') res = vA * s;
              else if (mathOp === '/') res = s !== 0 ? vA / s : 0;
            }
            return { ...row, [targetName]: Number(res.toFixed(4)) };
          }),
      });
      triggerToast(
        `تم اشتقاق وإضافة العمود الحسابي الجديد [${targetName}] بنجاح.`,
        `Engineered feature [${targetName}] successfully created.`
      );
      setNewColName('');
    } else if (feType === 'text_len') {
      applyOperation({
        type: 'feature_engineer',
        feType: 'text_len',
        column: targetName,
        mathColA,
        title_ar: `حساب طول النص من [${mathColA}] إلى العمود [${targetName}]`,
        title_en: `Compute text length from [${mathColA}] into [${targetName}]`,
        transformFn: (data) =>
          data.map((row) => ({
            ...row,
            [targetName]: String(row[mathColA] || '').length,
          })),
      });
      triggerToast(
        `تم حساب وإضافة طول النص في العمود [${targetName}] بنجاح.`,
        `Computed text length into feature [${targetName}] successfully.`
      );
      setNewColName('');
    } else if (feType === 'word_count') {
      applyOperation({
        type: 'feature_engineer',
        feType: 'word_count',
        column: targetName,
        mathColA,
        title_ar: `حساب عدد الكلمات في [${mathColA}] إلى العمود [${targetName}]`,
        title_en: `Compute word count from [${mathColA}] into [${targetName}]`,
        transformFn: (data) =>
          data.map((row) => ({
            ...row,
            [targetName]: String(row[mathColA] || '').trim().split(/\s+/).filter(Boolean).length,
          })),
      });
      triggerToast(
        `تم حساب وإضافة عدد الكلمات في العمود [${targetName}] بنجاح.`,
        `Computed word count into feature [${targetName}] successfully.`
      );
      setNewColName('');
    } else if (feType === 'text_concat') {
      applyOperation({
        type: 'feature_engineer',
        feType: 'text_concat',
        column: targetName,
        mathColA,
        mathColB,
        title_ar: `دمج النصوص من [${mathColA}] و [${mathColB}] إلى العمود [${targetName}]`,
        title_en: `Concatenated text from [${mathColA}] and [${mathColB}] into [${targetName}]`,
        transformFn: (data) =>
          data.map((row) => ({
            ...row,
            [targetName]: `${String(row[mathColA] || '')} ${String(row[mathColB] || '')}`.trim(),
          })),
      });
      triggerToast(
        `تم دمج العمودين النصيين في العمود الجديد [${targetName}] بنجاح.`,
        `Concatenated text features into [${targetName}] successfully.`
      );
      setNewColName('');
    }
  };

  const handleDropColumn = (col) => {
    applyOperation({
      type: 'drop_column',
      column: col,
      title_ar: `حذف العمود (${col}) من مجموعة البيانات`,
      title_en: `Dropped column (${col}) from dataset`,
      transformFn: (data) =>
        data.map((row) => {
          const next = { ...row };
          delete next[col];
          return next;
        }),
    });
    triggerToast(
      `تم حذف العمود [${col}] بنجاح من جدول البيانات.`,
      `Column [${col}] successfully dropped from dataset.`
    );
  };

  const handleApplyJoin = () => {
    if (!leftKey || !rightKey || secondaryDataset.length === 0) return;

    applyOperation({
      type: 'join',
      joinType,
      leftKey,
      rightKey,
      title_ar: `دمج علائقي (${joinType.toUpperCase()}) مع (${leftKey} = ${rightKey})`,
      title_en: `Relational ${joinType.toUpperCase()} Join on (${leftKey} = ${rightKey})`,
      transformFn: (data) => joinDatasets(data, secondaryDataset, leftKey, rightKey, joinType),
    });
    triggerToast(
      `تم دمج الجدول العلائقي بنجاح بأسلوب ${joinType.toUpperCase()}.`,
      `Relational ${joinType.toUpperCase()} join successfully executed.`
    );
  };

  const handleApplyGroupBy = () => {
    if (!groupCol || !aggCol) return;

    applyOperation({
      type: 'groupby',
      groupCol,
      aggCol,
      aggFn,
      title_ar: `تجميع البيانات حسب (${groupCol}) وحساب ${aggFn} للعمود (${aggCol})`,
      title_en: `Group by (${groupCol}) with ${aggFn}(${aggCol})`,
      transformFn: (data) =>
        groupByAggregate(data, groupCol, [{ column: aggCol, fn: aggFn }]),
    });
    triggerToast(
      `تم تجميع البيانات بنجاح حسب العمود [${groupCol}].`,
      `Dataset successfully aggregated by [${groupCol}].`
    );
  };

  const handleApplySort = () => {
    if (!sortCol) return;

    applyOperation({
      type: 'sort',
      column: sortCol,
      sortCol,
      sortOrder,
      title_ar: `فرز البيانات حسب العمود (${sortCol}) (${sortOrder === 'asc' ? 'تصاعدي' : 'تنازلي'})`,
      title_en: `Sorted dataset by (${sortCol}) ${sortOrder.toUpperCase()}`,
      transformFn: (data) =>
        [...data].sort((a, b) => {
          const valA = a[sortCol];
          const valB = b[sortCol];
          if (!isNaN(Number(valA)) && !isNaN(Number(valB))) {
            return sortOrder === 'asc' ? Number(valA) - Number(valB) : Number(valB) - Number(valA);
          }
          return sortOrder === 'asc'
            ? String(valA).localeCompare(String(valB))
            : String(valB).localeCompare(String(valA));
        }),
    });
    triggerToast(
      `تم إعادة فرز الصفوف بنجاح حسب العمود [${sortCol}].`,
      `Rows successfully sorted by [${sortCol}].`
    );
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      {/* Visual Action Toast Banner */}
      {actionToast && (
        <div
          style={{
            padding: '0.85rem 1.25rem',
            borderRadius: 'var(--radius-md, 8px)',
            background: 'var(--accent-teal-glow)',
            border: '1px solid var(--accent-teal)',
            color: 'var(--accent-teal)',
            fontSize: '0.88rem',
            fontWeight: 800,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            boxShadow: 'var(--shadow-sm)',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem' }}>
            <span className="badge badge-teal" style={{ fontSize: '0.75rem', display: 'inline-flex', alignItems: 'center', gap: '0.35rem' }}>
              <Check size={13} />
              <span>{isAr ? 'تم تنفيذ المعالجة' : 'Operation Applied'}</span>
            </span>
            <span>{actionToast}</span>
          </div>
          <button
            onClick={() => setActionToast(null)}
            style={{ background: 'none', border: 'none', color: 'var(--accent-teal)', cursor: 'pointer', display: 'inline-flex', alignItems: 'center' }}
          >
            <X size={15} />
          </button>
        </div>
      )}
      {/* Introduction Header */}
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
              <span className="badge badge-teal">{isAr ? 'المرحلة 4 من 5' : 'Phase 4 of 5'}</span>
              <span style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>
                <TermTooltip termKey="transformation" customLabel={isAr ? 'التحويل والدمج العلائقي' : 'Relational Transformation'} />
              </span>
            </div>
            <p style={{ margin: 0, fontSize: '0.9rem', color: 'var(--text-secondary)', lineHeight: 1.6, maxWidth: '850px' }}>
              {isAr
                ? 'أعِد تشكيل بياناتك لتناسب مهمتك: اختر الصفوف التي تريدها، أنشئ أعمدة جديدة من الموجودة، ادمج جدولين معاً، أو جمّع البيانات لاستخراج مقاييس ملخّصة.'
                : 'Reshape your data: filter rows, create new columns, merge tables, or aggregate values to prepare for model training.'}
            </p>
          </div>
          <div style={{ display: 'flex', gap: '0.4rem', flexWrap: 'wrap' }}>
            <TermTooltip termKey="feature_engineering" />
            <TermTooltip termKey="inner_join" />
          </div>
        </div>
      </div>

      {/* Sub-Operations Navigation Tabs */}
      <div
        style={{
          display: 'flex',
          gap: '0.5rem',
          borderBottom: '1px solid var(--border-subtle)',
          paddingBottom: '0.5rem',
          overflowX: 'auto',
        }}
      >
        {[
          { id: 'filter', code: '04.1', label_ar: 'فلترة الصفوف', label_en: 'Filter Rows', icon: Filter },
          { id: 'columns', code: '04.2', label_ar: 'هندسة الخصائص', label_en: 'Feature Engineering', icon: Columns },
          { id: 'join', code: '04.3', label_ar: 'دمج الجداول (Join)', label_en: 'Merge / Join', icon: GitMerge },
          { id: 'groupby', code: '04.4', label_ar: 'التجميع (Group-By)', label_en: 'Group-By', icon: Layers },
          { id: 'sort', code: '04.5', label_ar: 'الفرز', label_en: 'Sort', icon: ArrowUpDown },
        ].map((tab) => {
          const TabIcon = tab.icon;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '0.45rem',
                padding: '0.6rem 1.1rem',
                borderRadius: 'var(--radius-sm, 6px)',
                border: 'none',
                background: activeTab === tab.id ? 'var(--accent-teal)' : 'var(--bg-secondary)',
                color: activeTab === tab.id ? '#ffffff' : 'var(--text-secondary)',
                fontWeight: activeTab === tab.id ? '800' : '600',
                fontSize: '0.84rem',
                cursor: 'pointer',
                whiteSpace: 'nowrap',
                transition: 'all 0.15s ease',
              }}
            >
              <TabIcon size={14} />
              <span>{isAr ? tab.label_ar : tab.label_en}</span>
            </button>
          );
        })}
      </div>

      {/* Active Tab Panel */}
      <div
        style={{
          background: 'var(--bg-secondary)',
          border: '1px solid var(--border-subtle)',
          borderRadius: 'var(--radius-lg, 12px)',
          padding: '1.5rem',
          display: 'flex',
          flexDirection: 'column',
          gap: '1.25rem',
          boxShadow: 'var(--shadow-sm)',
        }}
      >
        {/* TAB 1: Multi-Condition Filtering */}
        {activeTab === 'filter' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
            <div>
              <h3 style={{ fontSize: '1.1rem', fontWeight: 800, color: 'var(--text-primary)', marginBottom: '0.35rem' }}>
                {isAr ? 'مُحرّك القواعد والإجراءات الشرطية (Rule-Based Engine)' : 'Rule-Based Engine & Row Processing'}
              </h3>
              <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', margin: 0 }}>
                {isAr
                  ? 'حدّد شروط منطقية لاختيار الصفوف، ثم اختر الإجراء المناسب: إبقاء الصفوف، حذفها، تحديث قيمة عمود، وسم الصفوف بعلم جديد، أو استبدال النص.'
                  : 'Define logical conditions to target rows, then execute actions: keep matching, drop matching, set column values, add flag tags, or replace text.'}
              </p>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', flexWrap: 'wrap' }}>
              <span style={{ fontSize: '0.85rem', fontWeight: 700, color: 'var(--text-secondary)' }}>
                {isAr ? 'الربط المنطقي بين الشروط:' : 'Condition Conjunction Logic:'}
              </span>
              <div style={{ display: 'inline-flex', borderRadius: 'var(--radius-sm, 6px)', border: '1px solid var(--border-medium)', overflow: 'hidden' }}>
                <button
                  onClick={() => setFilterLogic('AND')}
                  style={{
                    padding: '0.45rem 1rem',
                    border: 'none',
                    background: filterLogic === 'AND' ? 'var(--accent-teal)' : 'var(--bg-tertiary)',
                    color: filterLogic === 'AND' ? '#ffffff' : 'var(--text-secondary)',
                    fontSize: '0.8rem',
                    fontWeight: '800',
                    cursor: 'pointer',
                  }}
                >
                  AND ({isAr ? 'مطابقة جميع الشروط' : 'All Conditions'})
                </button>
                <button
                  onClick={() => setFilterLogic('OR')}
                  style={{
                    padding: '0.45rem 1rem',
                    border: 'none',
                    background: filterLogic === 'OR' ? 'var(--accent-teal)' : 'var(--bg-tertiary)',
                    color: filterLogic === 'OR' ? '#ffffff' : 'var(--text-secondary)',
                    fontSize: '0.8rem',
                    fontWeight: '800',
                    cursor: 'pointer',
                  }}
                >
                  OR ({isAr ? 'مطابقة أي شرط' : 'Any Condition'})
                </button>
              </div>
            </div>

            {/* Condition Rows */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
              {filterConditions.map((cond, idx) => (
                <div
                  key={cond.id}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.75rem',
                    padding: '0.75rem',
                    background: 'var(--bg-tertiary)',
                    borderRadius: 'var(--radius-sm, 6px)',
                    border: '1px solid var(--border-subtle)',
                    flexWrap: 'wrap',
                  }}
                >
                  <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)', width: '20px', fontFamily: 'var(--font-mono)' }}>
                    #{idx + 1}
                  </span>
                  <select
                    value={cond.column}
                    onChange={(e) => handleUpdateCondition(cond.id, 'column', e.target.value)}
                    style={{
                      padding: '0.45rem 0.75rem',
                      borderRadius: 'var(--radius-sm, 6px)',
                      border: '1px solid var(--border-medium)',
                      background: 'var(--bg-secondary)',
                      color: 'var(--text-primary)',
                      fontSize: '0.85rem',
                    }}
                  >
                    {columns.map((c) => (
                      <option key={c} value={c}>
                        {c}
                      </option>
                    ))}
                  </select>

                  <select
                    value={cond.op}
                    onChange={(e) => handleUpdateCondition(cond.id, 'op', e.target.value)}
                    style={{
                      padding: '0.45rem 0.75rem',
                      borderRadius: 'var(--radius-sm, 6px)',
                      border: '1px solid var(--border-medium)',
                      background: 'var(--bg-secondary)',
                      color: 'var(--text-primary)',
                      fontSize: '0.85rem',
                      fontFamily: 'var(--font-mono)',
                    }}
                  >
                    <option value="==">== ({isAr ? 'يساوي' : 'Equals'})</option>
                    <option value="!=">!= ({isAr ? 'لا يساوي' : 'Not Equals'})</option>
                    <option value=">">&gt; ({isAr ? 'أكبر من' : 'Greater than'})</option>
                    <option value=">=">&gt;= ({isAr ? 'أكبر أو يساوي' : 'Greater/Equal'})</option>
                    <option value="<">&lt; ({isAr ? 'أصغر من' : 'Less than'})</option>
                    <option value="<=">&lt;= ({isAr ? 'أصغر أو يساوي' : 'Less/Equal'})</option>
                    <option value="contains">contains ({isAr ? 'يحتوي النص' : 'Contains Text'})</option>
                    <option value="not_contains">not_contains ({isAr ? 'لا يحتوي النص' : 'Not Contains'})</option>
                    <option value="starts_with">starts_with ({isAr ? 'يبدأ بـ' : 'Starts With'})</option>
                    <option value="ends_with">ends_with ({isAr ? 'ينتهي بـ' : 'Ends With'})</option>
                    <option value="is_null">is_null ({isAr ? 'قيم مفقودة / فارغة' : 'Is Missing / Null'})</option>
                    <option value="not_null">not_null ({isAr ? 'قيم مكتملة / غير فارغة' : 'Is Not Null'})</option>
                    <option value="in_list">in_list ({isAr ? 'ضمن قائمة قيم (a,b,c)' : 'In List (a,b,c)'})</option>
                    <option value="between">between ({isAr ? 'محصور بين قيمتين (10,50)' : 'Between (10,50)'})</option>
                  </select>

                  {cond.op !== 'is_null' && cond.op !== 'not_null' && (
                    <input
                      type="text"
                      placeholder={
                        cond.op === 'in_list'
                          ? (isAr ? 'أدخل القيم مفصولة بفواصل: a, b, c' : 'Enter list: a, b, c')
                          : cond.op === 'between'
                          ? (isAr ? 'أدخل النطاق: min, max' : 'Enter range: min, max')
                          : (isAr ? 'أدخل القيمة...' : 'Enter value...')
                      }
                      value={cond.value}
                      onChange={(e) => handleUpdateCondition(cond.id, 'value', e.target.value)}
                      style={{
                        flex: 1,
                        minWidth: '150px',
                        padding: '0.45rem 0.75rem',
                        borderRadius: 'var(--radius-sm, 6px)',
                        border: '1px solid var(--border-medium)',
                        background: 'var(--bg-secondary)',
                        color: 'var(--text-primary)',
                        fontSize: '0.85rem',
                      }}
                    />
                  )}

                  {filterConditions.length > 1 && (
                    <button
                      onClick={() => handleRemoveCondition(cond.id)}
                      className="btn btn-danger btn-sm"
                      style={{ padding: '0.4rem 0.65rem', display: 'inline-flex', alignItems: 'center', gap: '0.35rem' }}
                      title={isAr ? 'حذف هذا الشرط' : 'Remove condition'}
                    >
                      <Trash2 size={13} />
                      <span>{isAr ? 'حذف' : 'Remove'}</span>
                    </button>
                  )}
                </div>
              ))}
            </div>

            {/* Rule Action Selector Section */}
            <div style={{ background: 'var(--bg-tertiary)', padding: '1rem 1.25rem', borderRadius: 'var(--radius-md, 8px)', border: '1px solid var(--border-medium)', display: 'flex', flexDirection: 'column', gap: '0.85rem' }}>
              <label style={{ fontSize: '0.88rem', fontWeight: 800, color: 'var(--text-primary)' }}>
                {isAr ? 'الإجراء المراد تنفيذه على الصفوف المطابقة:' : 'Action to Execute on Matching Rows:'}
              </label>

              <select
                value={ruleActionType}
                onChange={(e) => setRuleActionType(e.target.value)}
                style={{
                  width: '100%',
                  padding: '0.55rem 0.75rem',
                  borderRadius: 'var(--radius-sm, 6px)',
                  border: '1px solid var(--border-medium)',
                  background: 'var(--bg-secondary)',
                  color: 'var(--text-primary)',
                  fontSize: '0.85rem',
                  fontWeight: 700,
                }}
              >
                <option value="keep">{isAr ? 'إبقاء الصفوف المطابقة وحذف الباقي (Filter & Keep Matching)' : 'Keep Matching Rows (Drop Rest)'}</option>
                <option value="drop">{isAr ? 'حذف الصفوف المطابقة والإبقاء على الباقي (Drop Matching Rows)' : 'Drop Matching Rows (Keep Rest)'}</option>
                <option value="set_value">{isAr ? 'تحديث قيمة عمود في الصفوف المطابقة (Update / Set Value)' : 'Set Column Value on Match'}</option>
                <option value="flag">{isAr ? 'وسم / تمييز الصفوف بعلم جديد (Flag / Tag Matching Rows)' : 'Flag / Tag Matching Rows'}</option>
                <option value="replace_text">{isAr ? 'استبدال نص داخل العمود للصفوف المطابقة (Replace Substring)' : 'Replace Text on Match'}</option>
              </select>

              {/* Dynamic Action Configuration Parameters */}
              {ruleActionType === 'set_value' && (
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '0.75rem', paddingTop: '0.25rem' }}>
                  <div>
                    <label style={{ display: 'block', fontSize: '0.75rem', color: 'var(--text-secondary)', marginBottom: '0.25rem', fontWeight: 700 }}>
                      {isAr ? 'العمود المراد تحديثه:' : 'Target Column to Update:'}
                    </label>
                    <select
                      value={ruleTargetCol}
                      onChange={(e) => setRuleTargetCol(e.target.value)}
                      style={{
                        width: '100%',
                        padding: '0.45rem 0.75rem',
                        borderRadius: 'var(--radius-sm, 6px)',
                        border: '1px solid var(--border-medium)',
                        background: 'var(--bg-secondary)',
                        color: 'var(--text-primary)',
                        fontSize: '0.85rem',
                      }}
                    >
                      {columns.map((c) => (
                        <option key={c} value={c}>
                          {c}
                        </option>
                      ))}
                    </select>
                  </div>
                  <div>
                    <label style={{ display: 'block', fontSize: '0.75rem', color: 'var(--text-secondary)', marginBottom: '0.25rem', fontWeight: 700 }}>
                      {isAr ? 'القيمة الجديدة:' : 'New Value to Set:'}
                    </label>
                    <input
                      type="text"
                      placeholder={isAr ? 'أدخل القيمة الجديدة (مثال: neutral أو 0)' : 'e.g. neutral or 0'}
                      value={ruleSetValue}
                      onChange={(e) => setRuleSetValue(e.target.value)}
                      style={{
                        width: '100%',
                        padding: '0.45rem 0.75rem',
                        borderRadius: 'var(--radius-sm, 6px)',
                        border: '1px solid var(--border-medium)',
                        background: 'var(--bg-secondary)',
                        color: 'var(--text-primary)',
                        fontSize: '0.85rem',
                      }}
                    />
                  </div>
                </div>
              )}

              {ruleActionType === 'flag' && (
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(170px, 1fr))', gap: '0.75rem', paddingTop: '0.25rem' }}>
                  <div>
                    <label style={{ display: 'block', fontSize: '0.75rem', color: 'var(--text-secondary)', marginBottom: '0.25rem', fontWeight: 700 }}>
                      {isAr ? 'اسم عمود الوسم الجديد:' : 'New Flag Column Name:'}
                    </label>
                    <input
                      type="text"
                      placeholder="is_matched"
                      value={ruleFlagCol}
                      onChange={(e) => setRuleFlagCol(e.target.value)}
                      style={{
                        width: '100%',
                        padding: '0.45rem 0.75rem',
                        borderRadius: 'var(--radius-sm, 6px)',
                        border: '1px solid var(--border-medium)',
                        background: 'var(--bg-secondary)',
                        color: 'var(--text-primary)',
                        fontSize: '0.85rem',
                      }}
                    />
                  </div>
                  <div>
                    <label style={{ display: 'block', fontSize: '0.75rem', color: 'var(--text-secondary)', marginBottom: '0.25rem', fontWeight: 700 }}>
                      {isAr ? 'قيمة التطابق (Match):' : 'Value if Match:'}
                    </label>
                    <input
                      type="text"
                      placeholder="1"
                      value={ruleFlagVal}
                      onChange={(e) => setRuleFlagVal(e.target.value)}
                      style={{
                        width: '100%',
                        padding: '0.45rem 0.75rem',
                        borderRadius: 'var(--radius-sm, 6px)',
                        border: '1px solid var(--border-medium)',
                        background: 'var(--bg-secondary)',
                        color: 'var(--text-primary)',
                        fontSize: '0.85rem',
                      }}
                    />
                  </div>
                  <div>
                    <label style={{ display: 'block', fontSize: '0.75rem', color: 'var(--text-secondary)', marginBottom: '0.25rem', fontWeight: 700 }}>
                      {isAr ? 'قيمة عدم التطابق (Else):' : 'Value if No Match:'}
                    </label>
                    <input
                      type="text"
                      placeholder="0"
                      value={ruleElseVal}
                      onChange={(e) => setRuleElseVal(e.target.value)}
                      style={{
                        width: '100%',
                        padding: '0.45rem 0.75rem',
                        borderRadius: 'var(--radius-sm, 6px)',
                        border: '1px solid var(--border-medium)',
                        background: 'var(--bg-secondary)',
                        color: 'var(--text-primary)',
                        fontSize: '0.85rem',
                      }}
                    />
                  </div>
                </div>
              )}

              {ruleActionType === 'replace_text' && (
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(170px, 1fr))', gap: '0.75rem', paddingTop: '0.25rem' }}>
                  <div>
                    <label style={{ display: 'block', fontSize: '0.75rem', color: 'var(--text-secondary)', marginBottom: '0.25rem', fontWeight: 700 }}>
                      {isAr ? 'العمود النصي:' : 'Target Text Column:'}
                    </label>
                    <select
                      value={ruleTargetCol}
                      onChange={(e) => setRuleTargetCol(e.target.value)}
                      style={{
                        width: '100%',
                        padding: '0.45rem 0.75rem',
                        borderRadius: 'var(--radius-sm, 6px)',
                        border: '1px solid var(--border-medium)',
                        background: 'var(--bg-secondary)',
                        color: 'var(--text-primary)',
                        fontSize: '0.85rem',
                      }}
                    >
                      {columns.map((c) => (
                        <option key={c} value={c}>
                          {c}
                        </option>
                      ))}
                    </select>
                  </div>
                  <div>
                    <label style={{ display: 'block', fontSize: '0.75rem', color: 'var(--text-secondary)', marginBottom: '0.25rem', fontWeight: 700 }}>
                      {isAr ? 'النص المطلوب البحث عنه:' : 'Text to Find:'}
                    </label>
                    <input
                      type="text"
                      placeholder={isAr ? 'مثال: http://' : 'e.g. old_text'}
                      value={ruleFindText}
                      onChange={(e) => setRuleFindText(e.target.value)}
                      style={{
                        width: '100%',
                        padding: '0.45rem 0.75rem',
                        borderRadius: 'var(--radius-sm, 6px)',
                        border: '1px solid var(--border-medium)',
                        background: 'var(--bg-secondary)',
                        color: 'var(--text-primary)',
                        fontSize: '0.85rem',
                      }}
                    />
                  </div>
                  <div>
                    <label style={{ display: 'block', fontSize: '0.75rem', color: 'var(--text-secondary)', marginBottom: '0.25rem', fontWeight: 700 }}>
                      {isAr ? 'النص البديل:' : 'Replacement Text:'}
                    </label>
                    <input
                      type="text"
                      placeholder={isAr ? 'مثال: [URL]' : 'e.g. new_text'}
                      value={ruleReplaceText}
                      onChange={(e) => setRuleReplaceText(e.target.value)}
                      style={{
                        width: '100%',
                        padding: '0.45rem 0.75rem',
                        borderRadius: 'var(--radius-sm, 6px)',
                        border: '1px solid var(--border-medium)',
                        background: 'var(--bg-secondary)',
                        color: 'var(--text-primary)',
                        fontSize: '0.85rem',
                      }}
                    />
                  </div>
                </div>
              )}
            </div>

            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '0.5rem', flexWrap: 'wrap', gap: '0.5rem' }}>
              <button
                onClick={handleAddCondition}
                className="btn btn-secondary btn-sm"
                style={{ color: 'var(--accent-teal)', borderStyle: 'dashed', display: 'inline-flex', alignItems: 'center', gap: '0.4rem' }}
              >
                <Plus size={14} />
                <span>{isAr ? 'إضافة شرط جديد' : 'Add Condition'}</span>
              </button>

              <button
                onClick={handleApplyFilter}
                className="btn btn-primary"
                style={{ padding: '0.65rem 1.75rem', fontWeight: 800, display: 'inline-flex', alignItems: 'center', gap: '0.5rem' }}
              >
                <Play size={15} />
                <span>{isAr ? 'تنفيذ الإجراء الشرطي' : 'Execute Rule Action'}</span>
              </button>
            </div>
          </div>
        )}

        {/* TAB 2: Feature Engineering & Column Manager */}
        {activeTab === 'columns' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
            <div>
              <h3 style={{ fontSize: '1.1rem', fontWeight: 800, color: 'var(--text-primary)', marginBottom: '0.35rem' }}>
                {isAr ? 'هندسة الخصائص (Feature Engineering)' : 'Feature Engineering'}
              </h3>
              <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', margin: 0 }}>
                {isAr
                  ? 'أنشئ أعمدة جديدة من البيانات الموجودة لتحسين أداء النموذج. مثلاً: جمع عمودين رقمياً، أو حساب طول النص.'
                  : 'Create new columns from existing data to improve model performance — e.g. combine numeric columns or compute text length.'}
              </p>
            </div>

            {/* Create New Feature Form */}
            <div style={{ background: 'var(--bg-tertiary)', padding: '1.25rem', borderRadius: 'var(--radius-md, 8px)', border: '1px solid var(--border-subtle)', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
              <span style={{ fontSize: '0.88rem', fontWeight: 800, color: 'var(--text-primary)' }}>
                {isAr ? 'إنشاء عمود جديد:' : 'Create New Column:'}
              </span>

              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '0.75rem' }}>
                <div>
                  <label style={{ display: 'block', fontSize: '0.78rem', color: 'var(--text-secondary)', marginBottom: '0.3rem', fontWeight: 700 }}>
                    {isAr ? 'اسم العمود الجديد:' : 'New Feature Name:'}
                  </label>
                  <input
                    type="text"
                    placeholder={isAr ? 'مثال: total_score' : 'e.g. combined_ratio'}
                    value={newColName}
                    onChange={(e) => setNewColName(e.target.value)}
                    style={{
                      width: '100%',
                      padding: '0.45rem 0.75rem',
                      borderRadius: 'var(--radius-sm, 6px)',
                      border: '1px solid var(--border-medium)',
                      background: 'var(--bg-secondary)',
                      color: 'var(--text-primary)',
                      fontSize: '0.85rem',
                    }}
                  />
                </div>

                <div>
                  <label style={{ display: 'block', fontSize: '0.78rem', color: 'var(--text-secondary)', marginBottom: '0.3rem', fontWeight: 700 }}>
                    {isAr ? 'نوع الاشتقاق:' : 'Transformation Type:'}
                  </label>
                  <select
                    value={feType}
                    onChange={(e) => setFeType(e.target.value)}
                    style={{
                      width: '100%',
                      padding: '0.45rem 0.75rem',
                      borderRadius: 'var(--radius-sm, 6px)',
                      border: '1px solid var(--border-medium)',
                      background: 'var(--bg-secondary)',
                      color: 'var(--text-primary)',
                      fontSize: '0.85rem',
                    }}
                  >
                    <option value="math">{isAr ? 'عملية حسابية (+, -, *, /)' : 'Arithmetic Operation (+, -, *, /)'}</option>
                    <option value="text_len">{isAr ? 'طول النص بعدد الحروف (Character Length)' : 'Text Character Length'}</option>
                    <option value="word_count">{isAr ? 'عدد الكلمات في النص (Word Count)' : 'Text Word Count'}</option>
                    <option value="text_concat">{isAr ? 'دمج عمودين نصيين (Concatenate Text)' : 'Concatenate 2 Text Columns'}</option>
                  </select>
                </div>
              </div>

              {/* Dynamic Feature Derivation Controls */}
              <div style={{ background: 'var(--bg-secondary)', padding: '0.85rem 1rem', borderRadius: 'var(--radius-sm, 6px)', border: '1px solid var(--border-medium)' }}>
                {feType === 'math' && (
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', flexWrap: 'wrap' }}>
                    <div style={{ flex: 1, minWidth: '150px' }}>
                      <label style={{ display: 'block', fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '0.25rem', fontWeight: 700 }}>
                        {isAr ? 'العمود الأول (A):' : 'First Column (A):'}
                      </label>
                      <select
                        value={mathColA}
                        onChange={(e) => setMathColA(e.target.value)}
                        style={{
                          width: '100%',
                          padding: '0.45rem 0.75rem',
                          borderRadius: 'var(--radius-sm, 6px)',
                          border: '1px solid var(--border-medium)',
                          background: 'var(--bg-tertiary)',
                          color: 'var(--text-primary)',
                          fontSize: '0.85rem',
                        }}
                      >
                        {columns.map((c) => (
                          <option key={c} value={c}>
                            {c}
                          </option>
                        ))}
                      </select>
                    </div>

                    <div style={{ width: '80px' }}>
                      <label style={{ display: 'block', fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '0.25rem', fontWeight: 700 }}>
                        {isAr ? 'العملية:' : 'Operator:'}
                      </label>
                      <select
                        value={mathOp}
                        onChange={(e) => setMathOp(e.target.value)}
                        style={{
                          width: '100%',
                          padding: '0.45rem 0.5rem',
                          borderRadius: 'var(--radius-sm, 6px)',
                          border: '1px solid var(--border-medium)',
                          background: 'var(--bg-tertiary)',
                          color: 'var(--text-primary)',
                          fontSize: '0.85rem',
                          fontFamily: 'var(--font-mono)',
                          fontWeight: 800,
                        }}
                      >
                        <option value="+">+</option>
                        <option value="-">-</option>
                        <option value="*">×</option>
                        <option value="/">÷</option>
                      </select>
                    </div>

                    <div style={{ flex: 1, minWidth: '150px' }}>
                      <label style={{ display: 'block', fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '0.25rem', fontWeight: 700 }}>
                        {isAr ? 'المقابل (B):' : 'Second Term (B):'}
                      </label>
                      <select
                        value={mathMode}
                        onChange={(e) => setMathMode(e.target.value)}
                        style={{
                          width: '100%',
                          padding: '0.45rem 0.75rem',
                          borderRadius: 'var(--radius-sm, 6px)',
                          border: '1px solid var(--border-medium)',
                          background: 'var(--bg-tertiary)',
                          color: 'var(--text-primary)',
                          fontSize: '0.85rem',
                          marginBottom: '0.35rem',
                        }}
                      >
                        <option value="two_cols">{isAr ? 'عمود آخر' : 'Another Column'}</option>
                        <option value="scalar">{isAr ? 'رقم ثابت (Scalar)' : 'Constant Number'}</option>
                      </select>

                      {mathMode === 'two_cols' ? (
                        <select
                          value={mathColB}
                          onChange={(e) => setMathColB(e.target.value)}
                          style={{
                            width: '100%',
                            padding: '0.45rem 0.75rem',
                            borderRadius: 'var(--radius-sm, 6px)',
                            border: '1px solid var(--border-medium)',
                            background: 'var(--bg-tertiary)',
                            color: 'var(--text-primary)',
                            fontSize: '0.85rem',
                          }}
                        >
                          {columns.map((c) => (
                            <option key={c} value={c}>
                              {c}
                            </option>
                          ))}
                        </select>
                      ) : (
                        <input
                          type="number"
                          value={mathScalar}
                          onChange={(e) => setMathScalar(e.target.value)}
                          style={{
                            width: '100%',
                            padding: '0.45rem 0.75rem',
                            borderRadius: 'var(--radius-sm, 6px)',
                            border: '1px solid var(--border-medium)',
                            background: 'var(--bg-tertiary)',
                            color: 'var(--text-primary)',
                            fontSize: '0.85rem',
                          }}
                        />
                      )}
                    </div>
                  </div>
                )}

                {(feType === 'text_len' || feType === 'word_count') && (
                  <div>
                    <label style={{ display: 'block', fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '0.25rem', fontWeight: 700 }}>
                      {isAr ? 'عمود النص المستهدف:' : 'Target Text Column:'}
                    </label>
                    <select
                      value={mathColA}
                      onChange={(e) => setMathColA(e.target.value)}
                      style={{
                        width: '100%',
                        maxWidth: '320px',
                        padding: '0.45rem 0.75rem',
                        borderRadius: 'var(--radius-sm, 6px)',
                        border: '1px solid var(--border-medium)',
                        background: 'var(--bg-tertiary)',
                        color: 'var(--text-primary)',
                        fontSize: '0.85rem',
                      }}
                    >
                      {columns.map((c) => (
                        <option key={c} value={c}>
                          {c}
                        </option>
                      ))}
                    </select>
                  </div>
                )}

                {feType === 'text_concat' && (
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', flexWrap: 'wrap' }}>
                    <div style={{ flex: 1, minWidth: '160px' }}>
                      <label style={{ display: 'block', fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '0.25rem', fontWeight: 700 }}>
                        {isAr ? 'العمود النصي الأول:' : 'First Text Column:'}
                      </label>
                      <select
                        value={mathColA}
                        onChange={(e) => setMathColA(e.target.value)}
                        style={{
                          width: '100%',
                          padding: '0.45rem 0.75rem',
                          borderRadius: 'var(--radius-sm, 6px)',
                          border: '1px solid var(--border-medium)',
                          background: 'var(--bg-tertiary)',
                          color: 'var(--text-primary)',
                          fontSize: '0.85rem',
                        }}
                      >
                        {columns.map((c) => (
                          <option key={c} value={c}>
                            {c}
                          </option>
                        ))}
                      </select>
                    </div>

                    <div style={{ flex: 1, minWidth: '160px' }}>
                      <label style={{ display: 'block', fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '0.25rem', fontWeight: 700 }}>
                        {isAr ? 'العمود النصي الثاني:' : 'Second Text Column:'}
                      </label>
                      <select
                        value={mathColB}
                        onChange={(e) => setMathColB(e.target.value)}
                        style={{
                          width: '100%',
                          padding: '0.45rem 0.75rem',
                          borderRadius: 'var(--radius-sm, 6px)',
                          border: '1px solid var(--border-medium)',
                          background: 'var(--bg-tertiary)',
                          color: 'var(--text-primary)',
                          fontSize: '0.85rem',
                        }}
                      >
                        {columns.map((c) => (
                          <option key={c} value={c}>
                            {c}
                          </option>
                        ))}
                      </select>
                    </div>
                  </div>
                )}
              </div>

              <button
                onClick={handleAddFeature}
                disabled={!newColName.trim()}
                className="btn btn-primary"
                style={{ alignSelf: 'flex-start', padding: '0.55rem 1.5rem', fontWeight: 800, display: 'inline-flex', alignItems: 'center', gap: '0.5rem' }}
              >
                <Plus size={15} />
                <span>{isAr ? 'إضافة العمود الجديد' : 'Add New Column'}</span>
              </button>
            </div>

            {/* Existing Columns Drop / Manage List */}
            <div>
              <span style={{ fontSize: '0.85rem', fontWeight: 800, color: 'var(--text-primary)', display: 'block', marginBottom: '0.5rem' }}>
                {isAr ? 'الأعمدة الحالية (يمكنك حذف أي عمود لا تحتاجه):' : 'Current Columns (remove unneeded columns):'}
              </span>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
                {columns.map((col) => (
                  <div
                    key={col}
                    style={{
                      display: 'inline-flex',
                      alignItems: 'center',
                      gap: '0.5rem',
                      padding: '0.35rem 0.75rem',
                      borderRadius: 'var(--radius-sm, 6px)',
                      background: 'var(--bg-tertiary)',
                      border: '1px solid var(--border-subtle)',
                      fontSize: '0.8rem',
                      fontFamily: 'var(--font-mono)',
                    }}
                  >
                    <span>{col}</span>
                    {columns.length > 1 && (
                      <button
                        onClick={() => handleDropColumn(col)}
                        title={isAr ? `حذف العمود ${col}` : `Drop column ${col}`}
                        style={{
                          background: 'transparent',
                          border: 'none',
                          color: 'var(--accent-rose, #f43f5e)',
                          cursor: 'pointer',
                          padding: '0.15rem',
                          display: 'inline-flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          borderRadius: '4px',
                          transition: 'background 0.15s ease, color 0.15s ease',
                        }}
                        onMouseEnter={(e) => {
                          e.currentTarget.style.background = 'rgba(244, 63, 94, 0.15)';
                        }}
                        onMouseLeave={(e) => {
                          e.currentTarget.style.background = 'transparent';
                        }}
                      >
                        <X size={13} strokeWidth={2.5} />
                      </button>
                    )}
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* TAB 3: Relational Join */}
        {activeTab === 'join' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
            <div>
              <h3 style={{ fontSize: '1.1rem', fontWeight: 800, color: 'var(--text-primary)', marginBottom: '0.35rem' }}>
                {isAr ? 'دمج الجداول (Join / Merge)' : 'Merge / Join Tables'}
              </h3>
              <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', margin: 0 }}>
                {isAr
                  ? 'ادمج جدول بياناتك الحالي مع جدول آخر عبر عمود مشترك بينهما. اختر نوع الدمج المناسب لمهمتك.'
                  : 'Merge your current dataset with another table using a shared key column. Choose the join type that fits your needs.'}
              </p>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '1rem' }}>
              <div>
                <label style={{ display: 'block', fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '0.4rem', fontWeight: 700 }}>
                  {isAr ? 'مصدر المجموعة الثانية:' : 'Secondary Dataset Source:'}
                </label>
                <select
                  value={joinSource}
                  onChange={(e) => setJoinSource(e.target.value)}
                  style={{
                    width: '100%',
                    padding: '0.55rem 0.75rem',
                    background: 'var(--bg-tertiary)',
                    border: '1px solid var(--border-medium)',
                    borderRadius: 'var(--radius-sm, 6px)',
                    color: 'var(--text-primary)',
                    fontSize: '0.85rem',
                  }}
                >
                  <option value="sample">{isAr ? 'من العينات المرجعية (Benchmarks)' : 'From Academic Benchmarks'}</option>
                  <option value="library">{isAr ? 'من مكتبة الاستوديو المحفوظة' : 'From Saved Studio Library'}</option>
                </select>
              </div>

              <div>
                <label style={{ display: 'block', fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '0.4rem', fontWeight: 700 }}>
                  {isAr ? 'اختر الجدول الثاني للدمج:' : 'Select Secondary Table:'}
                </label>
                {joinSource === 'sample' ? (
                  <select
                    value={selectedSampleId}
                    onChange={(e) => setSelectedSampleId(e.target.value)}
                    style={{
                      width: '100%',
                      padding: '0.55rem 0.75rem',
                      background: 'var(--bg-tertiary)',
                      border: '1px solid var(--border-medium)',
                      borderRadius: 'var(--radius-sm, 6px)',
                      color: 'var(--text-primary)',
                      fontSize: '0.85rem',
                    }}
                  >
                    {sampleDatasetsCatalog.map((s) => (
                      <option key={s.id} value={s.id}>
                        {isAr ? s.name_ar : s.name_en} ({s.rows} {isAr ? 'صفاً' : 'rows'})
                      </option>
                    ))}
                  </select>
                ) : (
                  <select
                    value={selectedLibId}
                    onChange={(e) => setSelectedLibId(e.target.value)}
                    style={{
                      width: '100%',
                      padding: '0.55rem 0.75rem',
                      background: 'var(--bg-tertiary)',
                      border: '1px solid var(--border-medium)',
                      borderRadius: 'var(--radius-sm, 6px)',
                      color: 'var(--text-primary)',
                      fontSize: '0.85rem',
                    }}
                  >
                    {savedLibrary.length > 0 ? (
                      savedLibrary.map((lib) => (
                        <option key={lib.id} value={lib.id}>
                          {lib.name} ({lib.data ? lib.data.length : 0} {isAr ? 'صفاً' : 'rows'})
                        </option>
                      ))
                    ) : (
                      <option value="">{isAr ? 'لا توجد مجموعات محفوظة في المكتبة' : 'No saved datasets in library'}</option>
                    )}
                  </select>
                )}
              </div>

              <div>
                <label style={{ display: 'block', fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '0.4rem', fontWeight: 700 }}>
                  {isAr ? 'نوع الدمج (Join Type):' : 'Join Scheme:'}
                </label>
                <select
                  value={joinType}
                  onChange={(e) => setJoinType(e.target.value)}
                  style={{
                    width: '100%',
                    padding: '0.55rem 0.75rem',
                    background: 'var(--bg-tertiary)',
                    border: '1px solid var(--border-medium)',
                    borderRadius: 'var(--radius-sm, 6px)',
                    color: 'var(--text-primary)',
                    fontSize: '0.85rem',
                  }}
                >
                  <option value="inner">Inner Join ({isAr ? 'الصفوف الموجودة في كلا الجدولين فقط' : 'Only matching rows'})</option>
                  <option value="left">Left Join ({isAr ? 'كل صفوف جدولك + ما يُطابق من الثاني' : 'All left + matched right'})</option>
                  <option value="right">Right Join ({isAr ? 'كل صفوف الجدول الثاني + ما يُطابق' : 'All right + matched left'})</option>
                  <option value="outer">Full Outer ({isAr ? 'كل الصفوف من الجدولين، مع فراغ للمفقود' : 'All rows, NULLs where unmatched'})</option>
                </select>

                <OptionTipBox optionKey={`join_${joinType}`} isAr={isAr} />
              </div>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '1rem' }}>
              <div>
                <label style={{ display: 'block', fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '0.4rem', fontWeight: 700 }}>
                  {isAr ? 'مفتاح الربط في الجدول الحالي:' : 'Left Key Column:'}
                </label>
                <select
                  value={leftKey}
                  onChange={(e) => setLeftKey(e.target.value)}
                  style={{
                    width: '100%',
                    padding: '0.55rem 0.75rem',
                    background: 'var(--bg-tertiary)',
                    border: '1px solid var(--border-medium)',
                    borderRadius: 'var(--radius-sm, 6px)',
                    color: 'var(--text-primary)',
                    fontSize: '0.85rem',
                  }}
                >
                  {columns.map((c) => (
                    <option key={c} value={c}>
                      {c}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label style={{ display: 'block', fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '0.4rem', fontWeight: 700 }}>
                  {isAr ? 'مفتاح الربط في الجدول الثاني:' : 'Right Key Column:'}
                </label>
                <select
                  value={rightKey}
                  onChange={(e) => setRightKey(e.target.value)}
                  style={{
                    width: '100%',
                    padding: '0.55rem 0.75rem',
                    background: 'var(--bg-tertiary)',
                    border: '1px solid var(--border-medium)',
                    borderRadius: 'var(--radius-sm, 6px)',
                    color: 'var(--text-primary)',
                    fontSize: '0.85rem',
                  }}
                >
                  {secondaryColumns.length > 0 ? (
                    secondaryColumns.map((c) => (
                      <option key={c} value={c}>
                        {c}
                      </option>
                    ))
                  ) : (
                    <option value="">{isAr ? 'اختر مجموعة أولاً' : 'Select a secondary dataset first'}</option>
                  )}
                </select>
              </div>
            </div>

            <button
              onClick={handleApplyJoin}
              disabled={secondaryColumns.length === 0}
              className="btn btn-primary"
              style={{ alignSelf: 'flex-start', padding: '0.65rem 1.75rem', fontWeight: 800, display: 'inline-flex', alignItems: 'center', gap: '0.5rem' }}
            >
              <GitMerge size={15} />
              <span>{isAr ? 'تنفيذ الدمج' : 'Merge Tables'}</span>
            </button>
          </div>
        )}

        {/* TAB 4: GroupBy & Aggregate */}
        {activeTab === 'groupby' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
            <div>
              <h3 style={{ fontSize: '1.1rem', fontWeight: 800, color: 'var(--text-primary)', marginBottom: '0.35rem' }}>
                {isAr ? 'التجميع (Group-By)' : 'Group-By & Aggregate'}
              </h3>
              <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', margin: 0 }}>
                {isAr
                  ? 'جمّع الصفوف التي تشترك في نفس القيمة، ثم احسب مقياساً ملخّصاً (مثل المتوسط أو المجموع) لعمود آخر.'
                  : 'Group rows sharing the same value, then compute a summary statistic (mean, sum, etc.) for another column.'}
              </p>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem' }}>
              <div>
                <label style={{ display: 'block', fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '0.4rem', fontWeight: 700 }}>
                  {isAr ? 'جمّع الصفوف حسب عمود:' : 'Group By Column:'}
                </label>
                <select
                  value={groupCol}
                  onChange={(e) => setGroupCol(e.target.value)}
                  style={{
                    width: '100%',
                    padding: '0.55rem 0.75rem',
                    background: 'var(--bg-tertiary)',
                    border: '1px solid var(--border-medium)',
                    borderRadius: 'var(--radius-sm, 6px)',
                    color: 'var(--text-primary)',
                    fontSize: '0.85rem',
                  }}
                >
                  {columns.map((c) => (
                    <option key={c} value={c}>
                      {c}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label style={{ display: 'block', fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '0.4rem', fontWeight: 700 }}>
                  {isAr ? 'احسب مقياساً للعمود:' : 'Target Measure Column:'}
                </label>
                <select
                  value={aggCol}
                  onChange={(e) => setAggCol(e.target.value)}
                  style={{
                    width: '100%',
                    padding: '0.55rem 0.75rem',
                    background: 'var(--bg-tertiary)',
                    border: '1px solid var(--border-medium)',
                    borderRadius: 'var(--radius-sm, 6px)',
                    color: 'var(--text-primary)',
                    fontSize: '0.85rem',
                  }}
                >
                  {columns.map((c) => (
                    <option key={c} value={c}>
                      {c}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label style={{ display: 'block', fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '0.4rem', fontWeight: 700 }}>
                  {isAr ? 'الدالة الإحصائية:' : 'Aggregation Function:'}
                </label>
                <select
                  value={aggFn}
                  onChange={(e) => setAggFn(e.target.value)}
                  style={{
                    width: '100%',
                    padding: '0.55rem 0.75rem',
                    background: 'var(--bg-tertiary)',
                    border: '1px solid var(--border-medium)',
                    borderRadius: 'var(--radius-sm, 6px)',
                    color: 'var(--text-primary)',
                    fontSize: '0.85rem',
                  }}
                >
                  <option value="mean">{isAr ? 'المتوسط الحسابي (Mean)' : 'Mean (Average)'}</option>
                  <option value="sum">{isAr ? 'المجموع الإجمالي (Sum)' : 'Sum (Total)'}</option>
                  <option value="min">{isAr ? 'القيمة الصغرى (Min)' : 'Minimum'}</option>
                  <option value="max">{isAr ? 'القيمة العظمى (Max)' : 'Maximum'}</option>
                  <option value="std">{isAr ? 'الانحراف المعياري (Std)' : 'Standard Deviation'}</option>
                </select>
              </div>
            </div>

            <button
              onClick={handleApplyGroupBy}
              className="btn btn-primary"
              style={{ alignSelf: 'flex-start', padding: '0.65rem 1.75rem', fontWeight: 800, display: 'inline-flex', alignItems: 'center', gap: '0.5rem' }}
            >
              <Layers size={15} />
              <span>{isAr ? 'تطبيق التجميع والحساب' : 'Apply GroupBy Aggregation'}</span>
            </button>
          </div>
        )}

        {/* TAB 5: Sort */}
        {activeTab === 'sort' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
            <div>
              <h3 style={{ fontSize: '1.1rem', fontWeight: 800, color: 'var(--text-primary)', marginBottom: '0.35rem' }}>
                {isAr ? 'الفرز والترتيب' : 'Sort'}
              </h3>
              <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', margin: 0 }}>
                {isAr
                  ? 'رتّب صفوف البيانات تصاعدياً أو تنازلياً حسب أي عمود.'
                  : 'Sort your data rows in ascending or descending order by any column.'}
              </p>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '1rem' }}>
              <div>
                <label style={{ display: 'block', fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '0.4rem', fontWeight: 700 }}>
                  {isAr ? 'عمود الفرز:' : 'Sort Key Column:'}
                </label>
                <select
                  value={sortCol}
                  onChange={(e) => setSortCol(e.target.value)}
                  style={{
                    width: '100%',
                    padding: '0.55rem 0.75rem',
                    background: 'var(--bg-tertiary)',
                    border: '1px solid var(--border-medium)',
                    borderRadius: 'var(--radius-sm, 6px)',
                    color: 'var(--text-primary)',
                    fontSize: '0.85rem',
                  }}
                >
                  {columns.map((c) => (
                    <option key={c} value={c}>
                      {c}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label style={{ display: 'block', fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '0.4rem', fontWeight: 700 }}>
                  {isAr ? 'اتجاه الترتيب:' : 'Sort Direction:'}
                </label>
                <select
                  value={sortOrder}
                  onChange={(e) => setSortOrder(e.target.value)}
                  style={{
                    width: '100%',
                    padding: '0.55rem 0.75rem',
                    background: 'var(--bg-tertiary)',
                    border: '1px solid var(--border-medium)',
                    borderRadius: 'var(--radius-sm, 6px)',
                    color: 'var(--text-primary)',
                    fontSize: '0.85rem',
                  }}
                >
                  <option value="asc">{isAr ? 'تصاعدي (Ascending: من الأصغر للأكبر)' : 'Ascending (A-Z, 0-9)'}</option>
                  <option value="desc">{isAr ? 'تنازلي (Descending: من الأكبر للأصغر)' : 'Descending (Z-A, 9-0)'}</option>
                </select>
              </div>
            </div>

            <button
              onClick={handleApplySort}
              className="btn btn-primary"
              style={{ alignSelf: 'flex-start', padding: '0.65rem 1.75rem', fontWeight: 800, display: 'inline-flex', alignItems: 'center', gap: '0.5rem' }}
            >
              <ArrowUpDown size={15} />
              <span>{isAr ? 'تطبيق الفرز' : 'Sort Data'}</span>
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
