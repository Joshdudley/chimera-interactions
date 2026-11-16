# ChimeraX multi-model interface interaction commands.
# Examples:
#   add_hydrophobic #21.1/A #21.3/B-F printFile no #25.1.1 showLabels no
#   add_hbond       #21.1/A #21.3/B   printFile no #25.1.2 showLabels no
# Both endpoints may use hierarchical model IDs and a single chain or letter range.
# Example: add_hydrophobic #21.1/A-C #21.3/D-F printFile no #25.1.1 showLabels no

import os, re, numpy as np
from chimerax.core.commands import run, register, CmdDesc, StringArg
from chimerax.core.models import Model

HYDRO={"ALA","VAL","LEU","ILE","MET","PRO","PHE","TRP","TYR"}
AROM={"PHE","TYR","TRP","HIS"}
DONORS={"ARG":{"NE","NH1","NH2"},"ASN":{"ND2"},"GLN":{"NE2"},"HIS":{"ND1","NE2"},"LYS":{"NZ"},"SER":{"OG"},"THR":{"OG1"},"TRP":{"NE1"},"TYR":{"OH"},"CYS":{"SG"}}
ACCEPTORS={"ASN":{"OD1"},"ASP":{"OD1","OD2"},"GLN":{"OE1"},"GLU":{"OE1","OE2"},"HIS":{"ND1","NE2"},"SER":{"OG"},"THR":{"OG1"},"TYR":{"OH"},"CYS":{"SG"},"MET":{"SD"}}
ACID={"ASP":{"OD1","OD2"},"GLU":{"OE1","OE2"}}
BASE={"LYS":{"NZ"},"ARG":{"NE","NH1","NH2"}}
POLAR_RES={"ASN","GLN","SER","THR","ASP","GLU","ARG","LYS","HIS","CYS","TYR","TRP","MET"}
RING={"PHE":{"CG","CD1","CD2","CE1","CE2","CZ"},"TYR":{"CG","CD1","CD2","CE1","CE2","CZ"},"TRP":{"CD2","CE2","CE3","CZ2","CZ3","CH2"},"HIS":{"CG","ND1","CD2","CE1","NE2"}}


def _all_models(session):
    seen=set(); out=[]
    def walk(ms):
        for m in ms:
            if id(m) in seen: continue
            seen.add(id(m)); out.append(m)
            try: walk(list(m.child_models()))
            except Exception: pass
    walk(list(session.models)); return out


def _find(session, mid):
    wanted=tuple(int(x) for x in mid.split('.'))
    for m in _all_models(session):
        if tuple(getattr(m,'id',()))==wanted: return m
    return None


def _chains(text):
    if '-' not in text: return [text]
    p=text.split('-')
    if len(p)!=2 or len(p[0])!=1 or len(p[1])!=1: raise ValueError('Chain range must look like B-F')
    a,b=map(ord,p); step=1 if b>=a else -1
    return [chr(x) for x in range(a,b+step,step)]


def _endpoint(session, spec, allow_range=True):
    m=re.fullmatch(r'#(\d+(?:\.\d+)*)/([^/]+)',str(spec).strip())
    if not m: raise ValueError("Endpoint must look like #21.1/A or #21.3/B-F")
    model=_find(session,m.group(1))
    if model is None or not hasattr(model,'atoms'): raise ValueError(f'Could not locate coordinate model #{m.group(1)}')
    cs=_chains(m.group(2)) if allow_range else [m.group(2)]
    return model,m.group(1),cs


def _scene(a): return np.asarray(a.scene_coord,dtype=float)
def _dist(a,b): return float(np.linalg.norm(_scene(a)-_scene(b)))
def _spec(a): return a.atomspec

def _bool(v):
    s=str(v).lower()
    if s in ('yes','true','on','1'): return True
    if s in ('no','false','off','0'): return False
    raise ValueError('Expected yes/no, true/false, on/off, or 1/0')

def _donor(a): return (a.name=='N' and a.residue.name!='PRO') or a.name in DONORS.get(a.residue.name,set())
def _acceptor(a): return a.name=='O' or a.name in ACCEPTORS.get(a.residue.name,set())
def _salt_atoms(a,b):
    return ((a.name in ACID.get(a.residue.name,set()) and b.name in BASE.get(b.residue.name,set())) or
            (b.name in ACID.get(b.residue.name,set()) and a.name in BASE.get(a.residue.name,set())))

def _eligible_polar(a):
    return a.element.name in {'O','N','S'} and (a.residue.name in POLAR_RES or a.name in {'N','O','OXT'})

def _ring_atoms(r): return [a for a in r.atoms if a.name in RING.get(r.name,set())]
def _centroid_scene(r):
    aa=_ring_atoms(r)
    return None if not aa else np.mean(np.array([a.scene_coord for a in aa]),axis=0)

def _parent(session, dest, name):
    if not re.fullmatch(r'#\d+(?:\.\d+)*',dest): raise ValueError('Destination must be a model ID such as #25.1.1')
    old=_find(session,dest[1:])
    if old is not None:
        if hasattr(old,'atoms'): raise ValueError(f'{dest} is occupied by an atomic model')
        run(session,f'close {dest}')
    p=Model(name,session); session.models.add([p]); run(session,f'rename #{".".join(map(str,p.id))} name "{name}" id {dest}')
    return _find(session,dest[1:])

def _hide_labels(group):
    try: children=list(group.all_models())[1:]
    except Exception: children=[]
    for c in children:
        if 'distance' in str(getattr(c,'name','')).lower() or 'label' in str(getattr(c,'name','')).lower(): c.display=False

def _make_group(session, parent_spec, idx, name, pairs, color, show_labels):
    child=f'{parent_spec}.{idx}'
    # clear stale global group by name
    g=session.pb_manager.get_group(name,create=False)
    if g is not None and len(g.pseudobonds): g.pseudobonds.delete()
    for a,b in pairs:
        run(session,f'pbond {_spec(a)} {_spec(b)} color {color} radius 0.2 dashes 8 showDist true name "{name}" global',log=False)
    g=session.pb_manager.get_group(name,create=True)
    if not any(x is g for x in _all_models(session)): session.models.add([g])
    session.models.assign_id(g,tuple(int(x) for x in child[1:].split('.')))
    if not show_labels: _hide_labels(g)
    g.display=False
    return g

def _best_by_residue(candidates, cutoff):
    best={}
    for a,b,d in candidates:
        if d>cutoff: continue
        key=(id(a.residue),id(b.residue))
        if key not in best or d<best[key][2]: best[key]=(a,b,d)
    return list(best.values())

def _detect(kind,m1,c1,m2,c2):
    a1=[a for a in m1.atoms if a.residue.chain_id==c1]
    a2=[a for a in m2.atoms if a.residue.chain_id==c2]
    cand=[]
    if kind=='hydrophobic':
        x=[a for a in a1 if a.residue.name in HYDRO and a.element.name=='C' and a.name!='C']; y=[a for a in a2 if a.residue.name in HYDRO and a.element.name=='C' and a.name!='C']
        cand=[(a,b,_dist(a,b)) for a in x for b in y]; return [(a,b) for a,b,d in _best_by_residue(cand,4.5)]
    if kind=='bad_hydrophobic':
        h1=[a for a in a1 if a.residue.name in HYDRO and a.element.name=='C' and a.name!='C']; p1=[a for a in a1 if a.element.name in {'O','N'}]
        h2=[a for a in a2 if a.residue.name in HYDRO and a.element.name=='C' and a.name!='C']; p2=[a for a in a2 if a.element.name in {'O','N'}]
        cand=[(a,b,_dist(a,b)) for a in h1 for b in p2]+[(a,b,_dist(a,b)) for a in p1 for b in h2]
        return [(a,b) for a,b,d in _best_by_residue(cand,3.5)]
    if kind=='salt_bridge':
        x=[a for a in a1 if a.name in ACID.get(a.residue.name,set())]; xb=[a for a in a1 if a.name in BASE.get(a.residue.name,set())]
        y=[a for a in a2 if a.name in BASE.get(a.residue.name,set())]; ya=[a for a in a2 if a.name in ACID.get(a.residue.name,set())]
        cand=[(a,b,_dist(a,b)) for a in x for b in y]+[(a,b,_dist(a,b)) for a in xb for b in ya]
        return [(a,b) for a,b,d in _best_by_residue(cand,4.0)]
    if kind=='hbond':
        d1=[a for a in a1 if _donor(a)]; ac1=[a for a in a1 if _acceptor(a)]; d2=[a for a in a2 if _donor(a)]; ac2=[a for a in a2 if _acceptor(a)]
        cand=[(a,b,_dist(a,b)) for a in d1 for b in ac2 if not _salt_atoms(a,b)]+[(a,b,_dist(a,b)) for a in ac1 for b in d2 if not _salt_atoms(a,b)]
        return [(a,b) for a,b,d in _best_by_residue(cand,3.5)]
    if kind=='polar':
        x=[a for a in a1 if _eligible_polar(a)]; y=[a for a in a2 if _eligible_polar(a)]
        for a in x:
            for b in y:
                d=_dist(a,b)
                if _salt_atoms(a,b) or (d<=3.5 and ((_donor(a) and _acceptor(b)) or (_donor(b) and _acceptor(a)))): continue
                cand.append((a,b,d))
        return [(a,b) for a,b,d in _best_by_residue(cand,4.0)]
    raise ValueError(kind)

COLORS={'hydrophobic':'black','bad_hydrophobic':'grey','hbond':'pink','salt_bridge':'orange','polar':'purple'}

def _command(kind):
    def f(session, endpoint1, endpoint2, print_file_keyword, print_file_value, interaction_model_id, showLabels='yes'):
        try:
            m1,id1,cs1=_endpoint(session,endpoint1); m2,id2,cs2=_endpoint(session,endpoint2)
            if str(print_file_keyword).lower() not in ('printfile','print_file','printf_file'): raise ValueError('Expected printFile yes|no')
            write=_bool(print_file_value); show=_bool(showLabels)
            _parent(session,str(interaction_model_id),f'Model_{id1.replace(".","_")}_{kind}')
            total=0; lines=[]; pair_count=0
            for c1 in cs1:
                for c2 in cs2:
                    pair_count += 1
                    pairs=_detect(kind,m1,c1,m2,c2); total+=len(pairs)
                    name=f'Model_{id1.replace(".","_")}_{c1}_to_{id2.replace(".","_")}_{c2}_{kind}'
                    _make_group(session,str(interaction_model_id),pair_count,name,pairs,COLORS[kind],show)
                    lines += [f'{_spec(a)} <-> {_spec(b)}' for a,b in pairs]
                    session.logger.info(f'[{"add_"+kind}] #{id1}/{c1} to #{id2}/{c2}: {len(pairs)} contacts')
            if write:
                fn=kind+'.txt'; open(fn,'w').write('\n'.join(lines)+'\n'); session.logger.info(f'Wrote {os.path.abspath(fn)}')
            session.logger.info(f'Created {interaction_model_id} with {pair_count} chain-pair subgroup(s); total contacts: {total}')
        except Exception as e:
            session.logger.error(f'[add_{kind}] Error: {e}')
    return f

# Pi uses centroid markers and is implemented separately because one endpoint can be a ring centroid.
def add_pi(session, endpoint1, endpoint2, print_file_keyword, print_file_value, interaction_model_id, showLabels='yes'):
    try:
        from chimerax.markers import MarkerSet
        m1,id1,cs1=_endpoint(session,endpoint1); m2,id2,cs2=_endpoint(session,endpoint2)
        write=_bool(print_file_value); show=_bool(showLabels)
        _parent(session,str(interaction_model_id),f'Model_{id1.replace(".","_")}_pi')
        out=[]; pair_count=0
        for c1 in cs1:
          for c2 in cs2:
            pair_count += 1
            residues1=list({id(a.residue):a.residue for a in m1.atoms if a.residue.chain_id==c1 and a.residue.name in AROM|{'ARG','LYS','MET','CYS'}}.values())
            residues2=list({id(a.residue):a.residue for a in m2.atoms if a.residue.chain_id==c2 and a.residue.name in AROM|{'ARG','LYS','MET','CYS'}}.values())
            ms=MarkerSet(session,name=f'pi_centroids_{id1}_{c1}_to_{id2}_{c2}'); session.models.add([ms]); ms.display=False
            pairs=[]; n=1
            def marker(coord):
                nonlocal n
                # Marker coordinates are scene coordinates; transform into marker model coordinates (identity on creation).
                z=ms.create_marker(coord,(255,165,0,255),0.05,id=n); n+=1; return z
            for r1 in residues1:
                for r2 in residues2:
                    cA=_centroid_scene(r1); cB=_centroid_scene(r2)
                    if cA is not None and cB is not None and np.linalg.norm(cA-cB)<=6.5:
                        pairs.append((marker(cA),marker(cB))); out.append(f'{id1}/{c1}:{r1.number} pi-pi {id2}/{c2}:{r2.number}'); continue
                    if cA is not None and r2.name in {'ARG','LYS','MET','CYS'}:
                        names=BASE.get(r2.name,{'SD'} if r2.name=='MET' else {'SG'} if r2.name=='CYS' else set()); aa=[a for a in r2.atoms if a.name in names]
                        if aa:
                            b=min(aa,key=lambda a:np.linalg.norm(cA-_scene(a)))
                            if np.linalg.norm(cA-_scene(b))<=5.5: pairs.append((marker(cA),b))
                    elif cB is not None and r1.name in {'ARG','LYS','MET','CYS'}:
                        names=BASE.get(r1.name,{'SD'} if r1.name=='MET' else {'SG'} if r1.name=='CYS' else set()); aa=[a for a in r1.atoms if a.name in names]
                        if aa:
                            a=min(aa,key=lambda x:np.linalg.norm(cB-_scene(x)))
                            if np.linalg.norm(cB-_scene(a))<=5.5: pairs.append((a,marker(cB)))
            name=f'Model_{id1.replace(".","_")}_{c1}_to_{id2.replace(".","_")}_{c2}_pi'
            _make_group(session,str(interaction_model_id),pair_count,name,pairs,'tan',show)
            session.logger.info(f'[add_pi] #{id1}/{c1} to #{id2}/{c2}: {len(pairs)} contacts')
        if write: open('pi.txt','w').write('\n'.join(out)+'\n')
    except Exception as e: session.logger.error(f'[add_pi] Error: {e}')

for kind in ('hydrophobic','bad_hydrophobic','hbond','salt_bridge','polar'):
    fn=_command(kind)
    desc=CmdDesc(required=[('endpoint1',StringArg),('endpoint2',StringArg),('print_file_keyword',StringArg),('print_file_value',StringArg),('interaction_model_id',StringArg)],keyword=[('showLabels',StringArg)],synopsis=f'Generate {kind} contacts across model/chain endpoints')
    register('add_'+kind,desc,fn,logger=session.logger)

pi_desc=CmdDesc(required=[('endpoint1',StringArg),('endpoint2',StringArg),('print_file_keyword',StringArg),('print_file_value',StringArg),('interaction_model_id',StringArg)],keyword=[('showLabels',StringArg)],synopsis='Generate pi contacts across model/chain endpoints')
register('add_pi',pi_desc,add_pi,logger=session.logger)
print('Registered multi-model add_hydrophobic, add_bad_hydrophobic, add_hbond, add_salt_bridge, add_polar, add_pi')
