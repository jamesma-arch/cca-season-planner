
# cca_planner_app_v3.py
import streamlit as st
import pandas as pd
import numpy as np
import re
from io import BytesIO
from datetime import datetime, time

st.set_page_config(page_title="CCA Season Planner Pro v3", layout="wide")
st.title("CCA Season Planner Pro v3")
st.markdown("""
**New in v3**
- Multiple staff per club (up to *Staff Needed*)
- Real time-range clash detection (overlaps, not just exact matches)
- “Auto‑allocate” button using availability + preferences + fairness
- Exports: Master + Handbook + SOCS import + Staff summary
""")

DAYS = ["Monday","Tuesday","Wednesday","Thursday","Friday"]

# ------------------------- PARSERS -------------------------

def parse_activities(file):
    raw = pd.read_excel(file, sheet_name=0, header=None)
    header_row_idx = 3
    header = raw.iloc[header_row_idx].tolist()
    df = raw.iloc[header_row_idx+1:].copy()
    df.columns = header

    activity_col = None
    for c in df.columns:
        if isinstance(c, str) and "ACTIVITIES" in c.upper():
            activity_col = c
            break
    if not activity_col:
        activity_col = df.columns[2]

    ren = {
        activity_col: "Activity Name",
        "Year(s)": "Years",
        "Price": "Price",
        "Price ": "Price",
        "Time": "Time",
        "Muster Point": "Muster Point",
        "Activity Room/ Space": "Room",
        "Activity Room/ Space ": "Room",
        "Backup Room": "Backup Room",
        "Parent Pick up Point": "Pick Up Point",
        "Staff Leader": "Staff Leader",
        "No. Needed": "Staff Needed",
        "Max No.": "Max Students",
        "Notes": "Notes",
        "Staff Support": "Staff Support",
        "Additional Staff Support": "Additional Staff Support",
    }
    df = df.rename(columns={k:v for k,v in ren.items() if k in df.columns})

    rows=[]
    current_day=None
    current_group=None
    for _, r in df.iterrows():
        row_text = " ".join([str(x) for x in r.values if str(x)!="nan"])
        up=row_text.upper()

        if "ACTIVITIES" in up and any(d.upper() in up for d in DAYS):
            m=re.search(r"(MONDAY|TUESDAY|WEDNESDAY|THURSDAY|FRIDAY)", up)
            if m: current_day=m.group(1).title()
            if "EYC" in up: current_group="EYC"
            elif "YEAR 1" in up: current_group="Year 1"
            elif "PRE PREP" in up or "PRE-PREP" in up: current_group="Pre-Prep"
            else: current_group="Other"
            continue

        name=str(r.get("Activity Name","")).strip()
        if name=="" or name.lower()=="nan": continue
        d=r.to_dict()
        d["Day"]=current_day
        d["Group"]=current_group
        rows.append(d)

    acts=pd.DataFrame(rows)
    if "Staff Needed" in acts.columns:
        acts["Staff Needed"]=pd.to_numeric(acts["Staff Needed"], errors="coerce").fillna(0).astype(int)
    else:
        acts["Staff Needed"]=0

    if "Max Students" in acts.columns:
        acts["Max Students"]=pd.to_numeric(acts["Max Students"], errors="coerce")
    for col in ["Years","Price","Time","Room","Staff Leader","Notes"]:
        if col not in acts.columns: acts[col]=np.nan

    # assignment slots
    if "Assigned Staff" not in acts.columns:
        acts["Assigned Staff"] = [[] for _ in range(len(acts))]

    return acts


def parse_staffing(file):
    df=pd.read_excel(file, sheet_name=0)
    teacher_cols=["Unnamed: 0","Form","Room","Form Teacher","Allocation","Activity 1","Activity 2",
                  "Monday","Tuesday","Wednesday","Thursday","Friday","Notes"]
    support_cols=["Unnamed: 0","Form","Room","Allocation.1","Activity 1.1","Activity 2.1",
                  "Monday.1","Tuesday.1","Wednesday.1","Thursday.1","Friday.1","Notes.1"]

    teach=df[teacher_cols].copy().rename(columns={
        "Unnamed: 0":"Phase","Form Teacher":"Staff Name","Allocation":"Alloc Count",
        "Activity 1":"Pref 1","Activity 2":"Pref 2"
    })
    teach["Role"]="Teacher"

    supp=df[support_cols].copy().rename(columns={
        "Unnamed: 0":"Phase","Allocation.1":"Alloc Count","Activity 1.1":"Pref 1","Activity 2.1":"Pref 2",
        "Monday.1":"Monday","Tuesday.1":"Tuesday","Wednesday.1":"Wednesday","Thursday.1":"Thursday","Friday.1":"Friday",
        "Notes.1":"Notes"
    })
    supp["Staff Name"]=supp["Form"].where(supp["Form"].notna(), "")
    supp["Role"]="Support"

    staff=pd.concat([teach,supp], ignore_index=True)

    for d in DAYS:
        staff[d]=staff[d].astype(str).str.strip()

    staff=staff[staff["Staff Name"].astype(str).str.strip().ne("")]
    staff["Staff Name"]=staff["Staff Name"].astype(str).str.strip()

    for c in ["Pref 1","Pref 2"]:
        if c not in staff.columns:
            staff[c]=""
        staff[c]=staff[c].astype(str).fillna("").str.strip()

    return staff


# ------------------------- TIME HELPERS -------------------------

def parse_time_range(tstr):
    """
    Accepts '14:50 - 15:50' or similar. Returns (start,end) as minutes since midnight.
    If cannot parse, return (None,None).
    """
    if not isinstance(tstr,str): return (None,None)
    m=re.findall(r"(\d{1,2}):(\d{2})", tstr)
    if len(m) >= 2:
        (h1,mi1),(h2,mi2)=m[0],m[1]
        s=int(h1)*60+int(mi1)
        e=int(h2)*60+int(mi2)
        if e < s: e = s  # safety
        return (s,e)
    return (None,None)

def overlaps(a,b):
    (s1,e1),(s2,e2)=a,b
    if None in (s1,e1,s2,e2): return False
    return max(s1,s2) < min(e1,e2)

# ------------------------- CLASH DETECTION -------------------------

def staff_clashes(acts):
    clashes=[]
    tmp=acts.copy()
    tmp["Range"]=tmp["Time"].apply(parse_time_range)
    for day, gday in tmp.groupby("Day"):
        for staff_name in sorted({s for lst in gday["Assigned Staff"] for s in lst}):
            sg = gday[gday["Assigned Staff"].apply(lambda L: staff_name in L)]
            if len(sg) <= 1: continue
            ranges=list(sg["Range"])
            names=list(sg["Activity Name"])
            # check pairwise overlap
            for i in range(len(ranges)):
                for j in range(i+1,len(ranges)):
                    if overlaps(ranges[i],ranges[j]):
                        clashes.append({
                            "Day":day,
                            "Staff":staff_name,
                            "Activity A":names[i],
                            "Activity B":names[j],
                            "Time A":sg.iloc[i]["Time"],
                            "Time B":sg.iloc[j]["Time"]
                        })
    return pd.DataFrame(clashes)

def room_clashes(acts):
    clashes=[]
    tmp=acts.copy()
    tmp["Range"]=tmp["Time"].apply(parse_time_range)
    for (day,room), gr in tmp.groupby(["Day","Room"]):
        if pd.isna(room) or str(room).strip()=="":
            continue
        if len(gr)<=1: continue
        ranges=list(gr["Range"])
        names=list(gr["Activity Name"])
        for i in range(len(ranges)):
            for j in range(i+1,len(ranges)):
                if overlaps(ranges[i],ranges[j]):
                    clashes.append({
                        "Day":day,"Room":room,
                        "Activity A":names[i],
                        "Activity B":names[j],
                        "Time A":gr.iloc[i]["Time"],
                        "Time B":gr.iloc[j]["Time"]
                    })
    return pd.DataFrame(clashes)

# ------------------------- AVAILABILITY -------------------------

def availability_by_day(staff):
    return {d: staff[staff[d]=="1"]["Staff Name"].tolist() for d in DAYS}

def current_load(acts):
    # overall count per staff
    load={}
    for lst in acts["Assigned Staff"]:
        for s in lst:
            load[s]=load.get(s,0)+1
    return load

def day_load(acts, day):
    load={}
    dacts=acts[acts["Day"]==day]
    for lst in dacts["Assigned Staff"]:
        for s in lst:
            load[s]=load.get(s,0)+1
    return load

def staff_is_free(acts, day, time_str, staff_name):
    rng=parse_time_range(time_str)
    for _, r in acts[acts["Day"]==day].iterrows():
        if staff_name in r["Assigned Staff"]:
            if overlaps(parse_time_range(r["Time"]), rng):
                return False
    return True

def pref_score(activity_name, staff_row):
    a=activity_name.lower()
    p1=staff_row.get("Pref 1","").lower()
    p2=staff_row.get("Pref 2","").lower()
    score=0
    if p1 and p1 in a: score += 2
    if p2 and p2 in a: score += 1
    return score

def auto_allocate(acts, staff):
    avail=availability_by_day(staff)
    overall=current_load(acts)

    staff_index=staff.set_index("Staff Name", drop=False)

    acts = acts.copy()
    for idx, row in acts.iterrows():
        need=int(row["Staff Needed"])
        assigned=list(row["Assigned Staff"]) if isinstance(row["Assigned Staff"],list) else []
        if len(assigned) >= need: 
            continue

        candidates = avail.get(row["Day"], [])
        # filter by free at that time and not already assigned here
        candidates=[c for c in candidates if c not in assigned and staff_is_free(acts,row["Day"],row["Time"],c)]

        # sort by preference then fairness (lowest load)
        def key(c):
            sr=staff_index.loc[c]
            return (-pref_score(row["Activity Name"], sr), overall.get(c,0))
        candidates=sorted(candidates, key=key)

        for c in candidates:
            if len(assigned) >= need: break
            assigned.append(c)
            overall[c]=overall.get(c,0)+1

        acts.at[idx,"Assigned Staff"]=assigned

    return acts


# ------------------------- HANDBOOK -------------------------

def handbook_tables(acts):
    out={}
    for day in DAYS:
        day_df=acts[acts["Day"]==day]
        if day_df.empty: continue
        for group, gdf in day_df.groupby("Group"):
            gdf=gdf.sort_values(["Time","Activity Name"])
            sel=gdf[["Activity Name","Years","Time","Price","Room","Max Students"]].copy()
            out[(day,group)]=sel
    return out

def to_excel(dfs):
    buffer=BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for sheet, df in dfs.items():
            df.to_excel(writer, sheet_name=sheet[:31], index=False)
    buffer.seek(0)
    return buffer


# ------------------------- UI -------------------------

colA, colB = st.columns(2)
with colA:
    acts_file = st.file_uploader("Upload Activities Season file", type=["xlsx"])
with colB:
    staff_file = st.file_uploader("Upload Staff Allocation file", type=["xlsx"])

if not (acts_file and staff_file):
    st.info("Upload both files to begin.")
    st.stop()

acts=parse_activities(acts_file)
staff=parse_staffing(staff_file)

avail=availability_by_day(staff)

# Demand vs supply
st.subheader("1) Demand vs supply")
demand = acts.groupby("Day")["Staff Needed"].sum().reindex(DAYS).fillna(0).astype(int)
supply = pd.Series({d:(staff[d]=="1").sum() for d in DAYS})
summary=pd.DataFrame({
    "Staff required":demand,
    "Staff available":supply,
    "Gap":supply-demand
})
st.dataframe(summary, use_container_width=True)

# Auto-allocate button
st.subheader("2) Auto‑allocate")
if st.button("Auto‑allocate remaining gaps (safe mode)"):
    acts = auto_allocate(acts, staff)
    st.success("Auto‑allocation complete. Scroll down to review.")

# Manual assignment
st.subheader("3) Manual assignment (multi‑staff per club)")

disp = acts.reset_index(drop=True)
filters = st.columns([1,1,2])
with filters[0]:
    f_day=st.selectbox("Day", ["All"]+DAYS)
with filters[1]:
    f_group=st.selectbox("Group", ["All"]+sorted(disp["Group"].dropna().unique().tolist()))
with filters[2]:
    f_search=st.text_input("Search")

if f_day!="All":
    disp=disp[disp["Day"]==f_day]
if f_group!="All":
    disp=disp[disp["Group"]==f_group]
if f_search.strip():
    s=f_search.lower()
    disp=disp[disp.apply(lambda r:any(s in str(v).lower() for v in r.values), axis=1)]

# Build UI rows
new_assigned = acts["Assigned Staff"].tolist()
for i, row in disp.iterrows():
    st.markdown(f"**{row['Day']} | {row['Time']} | {row['Activity Name']}**  \n"
                f"Group: {row['Group']} • Years: {row['Years']} • Room: {row['Room']} • "
                f"Needed: {row['Staff Needed']}")
    options=[""]+sorted(set(avail.get(row["Day"], [])))

    slots=[]
    for k in range(int(row["Staff Needed"])):
        current = row["Assigned Staff"][k] if k < len(row["Assigned Staff"]) else ""
        slots.append(st.selectbox(
            f"Staff slot {k+1}",
            options,
            index=options.index(current) if current in options else 0,
            key=f"slot_{i}_{k}"
        ))
    slots=[s for s in slots if s!=""]
    new_assigned[row.name]=slots
    st.divider()

acts["Assigned Staff"]=new_assigned

# Gaps
st.subheader("4) Gaps")
gaps=acts[acts["Assigned Staff"].apply(len) < acts["Staff Needed"]]
if gaps.empty:
    st.success("No gaps 🎉")
else:
    st.warning("Still missing staff for these clubs:")
    show=gaps[["Day","Group","Activity Name","Time","Room","Staff Needed"]].copy()
    show["Assigned"]=gaps["Assigned Staff"].apply(lambda L:", ".join(L))
    st.dataframe(show, use_container_width=True)

# Clashes
st.subheader("5) Clashes")
sc=staff_clashes(acts)
rc=room_clashes(acts)

c1,c2=st.columns(2)
with c1:
    st.markdown("**Staff overlaps**")
    st.dataframe(sc if not sc.empty else pd.DataFrame([{"Status":"None"}]), use_container_width=True)
with c2:
    st.markdown("**Room overlaps**")
    st.dataframe(rc if not rc.empty else pd.DataFrame([{"Status":"None"}]), use_container_width=True)

# Handbook preview
st.subheader("6) Parent handbook preview")
tabs=handbook_tables(acts)
for (day,group), df_tab in tabs.items():
    with st.expander(f"{day} — {group}", expanded=False):
        st.dataframe(df_tab, use_container_width=True)

# Staff summary export
st.subheader("7) Exports")
master = acts.copy()
master["Assigned Staff"] = master["Assigned Staff"].apply(lambda L:", ".join(L))

master_csv=master.to_csv(index=False).encode("utf-8")
st.download_button("Download Master CSV", master_csv, "master_with_assignments.csv", "text/csv")

# SOCS export (safe version, ignores missing columns)
socs_cols = ["Activity Name", "Years", "Day", "Time", "Price", "Room", "Max Students", "Assigned Staff"]

available_cols = [c for c in socs_cols if c in master.columns]

socs = master[available_cols].copy()
socs_csv = socs.to_csv(index=False).encode("utf-8")

st.download_button(
    "Download SOCS Import CSV (best-effort)",
    socs_csv,
    "socs_import_activities.csv",
    "text/csv"
)

# Excel pack
handbook_sheets={f"{day}_{group}":df for (day,group), df in tabs.items()}
excel = to_excel({"MASTER":master, **handbook_sheets})
st.download_button(
    "Download Excel Pack (Master + Handbook)",
    excel,
    "season_pack_v3.xlsx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

# Staff load summary
load = current_load(acts)
load_df=pd.DataFrame([{"Staff":k,"Clubs assigned":v} for k,v in sorted(load.items(), key=lambda x:-x[1])])
st.download_button(
    "Download Staff Load CSV",
    load_df.to_csv(index=False).encode("utf-8"),
    "staff_load.csv",
    "text/csv"
)
