"""Public regression metrics and multi-seed summaries."""
from __future__ import annotations
import numpy as np
import pandas as pd

def regression_metrics(frame):
    y=frame.nominal_concentration_mg_l.to_numpy(float); p=frame.predicted_concentration_mg_l.to_numpy(float); residual=p-y; denominator=np.sum((y-y.mean())**2)
    return {"coefficient_of_determination":float(1-np.sum(residual**2)/denominator) if denominator>0 else np.nan,"rmse_mg_l":float(np.sqrt(np.mean(residual**2))),"mae_mg_l":float(np.mean(np.abs(residual))),"bias_mg_l":float(np.mean(residual)),"median_absolute_error_mg_l":float(np.median(np.abs(residual))),"total_squared_error":float(np.sum(residual**2)),"observations":len(frame)}

def build_metrics(predictions):
    rows=[]
    for (scope,seed,model),g in predictions.groupby(["evaluation_scope","network_seed","model_name"],sort=False):
        base={"record_type":"seed_metric","evaluation_scope":scope,"network_seed":int(seed),"model_name":model}
        rows.append({**base,"scope_type":"pooled","scope_id":"all",**regression_metrics(g)})
        if scope=="specimen_grouped_cross_validation":
            for fold,x in g.groupby("fold"): rows.append({**base,"scope_type":"fold","scope_id":str(int(fold)),**regression_metrics(x)})
        for specimen,x in g.groupby("specimen_id"): rows.append({**base,"scope_type":"specimen","scope_id":specimen,**regression_metrics(x)})
        for stage,x in g.groupby("concentration_stage"): rows.append({**base,"scope_type":"concentration_stage","scope_id":f"C{int(stage)}",**regression_metrics(x)})
    detail=pd.DataFrame(rows); pooled=detail[(detail.scope_type=="pooled")]; summaries=[]
    for (scope,model),g in pooled.groupby(["evaluation_scope","model_name"]):
        for metric in ["coefficient_of_determination","rmse_mg_l","mae_mg_l","bias_mg_l"]:
            values=g[metric].to_numpy(float); summaries.append({"record_type":"seed_distribution","evaluation_scope":scope,"model_name":model,"scope_type":"across_network_seeds","scope_id":metric,"metric_name":metric,"seed_count":len(values),"mean":float(values.mean()),"sample_standard_deviation":float(values.std(ddof=1)) if len(values)>1 else np.nan,"median":float(np.median(values)),"minimum":float(values.min()),"maximum":float(values.max())})
    return pd.concat([detail,pd.DataFrame(summaries)],ignore_index=True,sort=False)

