"""Apply the frozen practical gate without tuning or selecting a candidate."""
import json,numpy as np
from phase7_design import *

def main():
    rows=json.loads((OUT/'control_results.json').read_text());g=P['primary_gate'];cells=[]
    def group(plant,task,method,budget):return [r for r in rows if r['plant']==plant and r['task']==task and r['kind']==g['family'] and r['method']==method and r['budget']==budget]
    for plant in P['plants']:
        for task in P['tasks']:
            means={m:float(np.mean([r['momentum_vector_rms_mean'] for r in group(plant,task,m,g['budget'])])) for m in P['methods']}
            baseline=min([m for m in P['methods'] if m!='spectral_A'],key=lambda m:means[m]);improvement=1-means['spectral_A']/means[baseline]
            paired=[]
            for w in P['world_seeds']:
                a=next(r['momentum_vector_rms_mean'] for r in group(plant,task,'spectral_A',g['budget']) if r['world_seed']==w)
                b=next(r['momentum_vector_rms_mean'] for r in group(plant,task,baseline,g['budget']) if r['world_seed']==w);paired.append(a<b)
            other=[]
            for budget in [16,256]:
                mm={m:float(np.mean([r['momentum_vector_rms_mean'] for r in group(plant,task,m,budget)])) for m in P['methods']}
                best=min(mm[m] for m in P['methods'] if m!='spectral_A');other.append({'budget':budget,'relative_degradation':mm['spectral_A']/best-1})
            passed=(improvement>=g['minimum_relative_mean_RMS_improvement'] and sum(paired)>=g['paired_seeds_improving_each_cell']
                     and max(x['relative_degradation'] for x in other)<=g['other_budget_worst_allowed_relative_degradation'])
            cells.append({'plant':plant,'task':task,'best_baseline':baseline,'method_mean_RMS':means,
                          'candidate_relative_improvement':improvement,'paired_seeds_improving':sum(paired),'other_budget_checks':other,'passed':bool(passed)})
    numerical=all(x['passed'] for x in cells)
    decision={'generic_algorithm_novelty_cleared':False,'primary_numerical_gate_passed':numerical,'cells':cells,
              'confirmation_required':numerical,'route':'controlled empirical follow-up; algorithm novelty still not cleared' if numerical else 'stop new-algorithm route; comparative-study validation and manuscript consolidation',
              'interpretation':'Practical descriptive screen, not a significance test. Comparator is best baseline, not random alone. No posthoc tuned variants.'}
    (OUT/'decision.json').write_text(json.dumps(decision,indent=2));print(json.dumps(decision,indent=2))
if __name__=='__main__':main()
