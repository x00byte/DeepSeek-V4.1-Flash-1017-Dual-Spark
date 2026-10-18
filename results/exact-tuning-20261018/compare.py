"""Validate matching measured requests and summarize the two exact profiles."""
from collections import defaultdict
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).parent
labels=('baseline-exact1536','candidate-exact-tuned')
runs={label:json.loads((ROOT/label/'run.json').read_text())for label in labels}
assert all(r['status']=='complete'for r in runs.values()),'both full screens must complete'
assert len({r['plan_sha256']for r in runs.values()})==1,'plans changed'
plan=json.loads((ROOT/'prepared/plan.json').read_text())
rows={label:[json.loads(line)for line in (ROOT/label/'requests.jsonl').read_text().splitlines()]for label in labels}
assert all(len(v)==44 for v in rows.values()),'44 timed requests required per profile'
assert all(row[key]is True for row in rows[labels[1]]for key in
           ('baseline_matched_prompt','baseline_token_ids_match','baseline_text_match')),'output or prompt changed'

summaries={label:json.loads((ROOT/label/'summary.json').read_text())for label in labels}
groups=defaultdict(dict)
for label in labels:
    for burst,summary in zip(plan['bursts'],summaries[label],strict=True):
        item=burst[0][0]
        key=(item['phase'],item['thinking'],item['concurrency'],item['expected_prompt_tokens'])
        if key not in groups or label not in groups[key]:groups[key][label]=[]
        assert summary['outside_clients']['verified']and summary['cap_satisfied']
        assert summary['pure_decode_verified']if item['phase']=='decode'else summary['cold_prefill_verified']
        groups[key][label].append(summary)

result=[]
for (phase,thinking,concurrency,inputs),arms in groups.items():
    metric='raw_decode_aggregate_tps'if phase=='decode'else'effective_cold_prefill_input_tps'
    values={label:[r[metric]for r in arms[label]]for label in labels}
    medians={label:statistics.median(v)for label,v in values.items()}
    entry=dict(phase=phase,thinking=thinking,concurrency=concurrency,input_tokens=inputs,
               metric=metric,values=values,medians=medians,
               median_gain_percent=100*(medians[labels[1]]/medians[labels[0]]-1))
    if phase=='decode':
        entry['skip_nograph']={label:[r['speculative_draft_deltas'].get('skip.nograph',0)for r in arms[label]]for label in labels}
        entry['target_eager']={label:[r['graph_stat_deltas']['target_graphs'].get('eager',0)for r in arms[label]]for label in labels}
        wall={}
        for label in labels:
            by_repeat=defaultdict(list)
            for row in rows[label]:
                if(row['phase'],row['thinking'],row['concurrency'],row['prompt_tokens'])==(phase,thinking,concurrency,inputs):
                    by_repeat[row['repeat']].append(row)
            wall[label]=[sum(r['completion_tokens']for r in cohort)/(max(r['ended_s']for r in cohort)-min(r['request_started_s']for r in cohort))for cohort in by_repeat.values()]
        entry['end_to_end_output_tps']=wall
    result.append(entry)
out=dict(schema=1,model=runs[labels[0]]['target_model'],revision=runs[labels[0]]['revision'],
         plan_sha256=runs[labels[0]]['plan_sha256'],timed_requests_per_profile=44,
         timed_output_tokens_per_profile={label:sum(r['completion_tokens']for r in records)for label,records in rows.items()},
         all_timed_prompt_ids_and_full_output_token_ids_and_text_match=True,
         baseline=dict(expert_block=1024,draft_graph_slots=4,target_graph_rows_max=32,target_graph_warm='decode'),
         candidate=dict(expert_block=1536,draft_graph_slots=8,target_graph_rows_max=64,target_graph_warm='all'),
         arithmetic='existing exact prefill; same weights, precision and model kernels',
         limitations=['two repeats per cohort, one boot per profile','64-token varied-prose decode; finite batch with finishing tails',
                      'cap-one prefill at1536/3072 tokens; includes scheduling and first output','bounded output reproducibility, not a broad quality evaluation',
                      'combined profile; no separate attribution to each setting'],comparisons=result)
(ROOT/'comparison.json').write_text(json.dumps(out,indent=2)+'\n')
for entry in result:
    print(json.dumps({k:v for k,v in entry.items()if k not in ('values','end_to_end_output_tps')}))
