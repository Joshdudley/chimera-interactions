"""Show/hide interaction groups created by chimerax_add_interactions_multimodel.py.

Examples:
  runscript /path/to/show_interactions_multimodel.py
  show_interactions #21.1/A #21.3/B display_Dist false
  show_interactions #21.1/A-C #21.3/B-F display_Dist false
  show_interactions #21.1/A #21.3/B keep #21.1/A:10,20 #21.3/B:30 display_Dist false
  hide_interactions #21.1/A #21.3/B

Both endpoints accept one chain or a letter range. The command displays existing
interaction pseudobonds; it does not re-run interaction detection.
"""
import re
import numpy as np
from chimerax.core.commands import CmdDesc, RestOfLine, register, run

GROUP = re.compile(r'^Model_(?P<m1>\d+(?:_\d+)*)_(?P<c1>[^_]+)_to_(?P<m2>\d+(?:_\d+)*)_(?P<c2>[^_]+)_(?P<kind>bad_hydrophobic|salt_bridge|hydrophobic|hbond|polar|pi)$')
ENDPOINT = re.compile(r'^#(?P<model>\d+(?:\.\d+)*)/(?P<chains>[^/\s]+)$')
KEEP = re.compile(r'^(#\d+(?:\.\d+)*)/([^/:]+):([\d,]+)$')


def all_models(session):
    seen=set(); out=[]
    def walk(items):
        for m in items:
            if id(m) in seen: continue
            seen.add(id(m)); out.append(m)
            try: walk(list(m.child_models()))
            except Exception: pass
    walk(list(session.models)); return out


def spec(model): return '#' + '.'.join(map(str, model.id))

def chains(text):
    if '-' not in text: return [text]
    p=text.split('-')
    if len(p)!=2 or len(p[0])!=1 or len(p[1])!=1:
        raise ValueError('Chain range must look like A-C or B-F.')
    a,b=map(ord,p); step=1 if b>=a else -1
    return [chr(x) for x in range(a,b+step,step)]

def endpoint(text):
    m=ENDPOINT.fullmatch(text)
    if not m: raise ValueError('Endpoint must look like #21.1/A or #21.3/B-F.')
    return m['model'], chains(m['chains'])

def find_structure(session, mid):
    wanted='#'+mid
    for m in all_models(session):
        if spec(m)==wanted and hasattr(m,'atoms'): return m
    return None

def parse(arguments):
    words=str(arguments).split()
    if len(words)<2: raise ValueError('Specify two endpoints, e.g. #21.1/A #21.3/B-F.')
    e1,e2=words.pop(0),words.pop(0)
    id1,cs1=endpoint(e1); id2,cs2=endpoint(e2)
    keeps=[]; labels=None; hide_others=True; in_keep=False
    i=0
    while i<len(words):
        w=words[i]
        low=w.lower()
        if low=='keep': in_keep=True
        elif low in ('display_dist','displaydist'):
            i+=1
            if i>=len(words) or words[i].lower() not in ('true','false','yes','no'):
                raise ValueError('display_Dist requires true or false.')
            labels=words[i].lower() in ('true','yes'); in_keep=False
        elif low in ('hide_other_chains','hideotherchains'):
            i+=1
            if i>=len(words) or words[i].lower() not in ('true','false','yes','no'):
                raise ValueError('hide_other_chains requires true or false.')
            hide_others=words[i].lower() in ('true','yes'); in_keep=False
        elif in_keep:
            km=KEEP.fullmatch(w)
            if not km: raise ValueError('Invalid keep residue specification: '+w)
            keeps.append((km[1][1:],km[2],{int(n) for n in km[3].split(',') if n}))
        else:
            raise ValueError('Unknown option: '+w)
        i+=1
    return id1,cs1,id2,cs2,keeps,labels,hide_others

def group_details(group):
    if not hasattr(group,'pseudobonds') or hasattr(group,'atoms'): return None
    m=GROUP.fullmatch(str(getattr(group,'name','')))
    if not m: return None
    return m['m1'].replace('_','.'),m['c1'],m['m2'].replace('_','.'),m['c2'],m['kind']

def label_models(group):
    try: children=list(group.all_models())[1:]
    except Exception:
        try: children=list(group.child_models())
        except Exception: children=[]
    return [c for c in children if ('distance' in str(getattr(c,'name','')).lower() or 'label' in str(getattr(c,'name','')).lower())]

def residue_for_atom(atom, structures, allowed):
    r=getattr(atom,'residue',None)
    if r is not None:
        for mid,model in structures.items():
            if getattr(r,'structure',None) is model and (mid,r.chain_id) in allowed:
                return r
    # pi centroid marker: infer the nearest unique residue in either requested endpoint.
    pos=np.asarray(getattr(atom,'scene_coord',getattr(atom,'coord',None)),dtype=float)
    candidates=[]
    for mid,model in structures.items():
        for rr in model.residues:
            if (mid,rr.chain_id) not in allowed: continue
            heavy=[a for a in rr.atoms if getattr(a.element,'name','')!='H']
            if not heavy: continue
            d=min(float(np.linalg.norm(pos-np.asarray(getattr(a,'scene_coord',a.coord),dtype=float))) for a in heavy)
            candidates.append((d,rr))
    candidates.sort(key=lambda x:x[0])
    if candidates and candidates[0][0]<=3.0 and (len(candidates)==1 or candidates[1][0]-candidates[0][0]>=0.15):
        return candidates[0][1]
    return None

def show_residues(session, structures, residues, endpoints):
    by_model_chain={}
    for r in residues:
        mid=next((k for k,v in structures.items() if v is getattr(r,'structure',None)),None)
        if mid: by_model_chain.setdefault((mid,r.chain_id),set()).add(r.number)
    chosen=[]
    for (mid,ch),nums in sorted(by_model_chain.items()):
        chosen.append(f'#{mid}/{ch}:{",".join(map(str,sorted(nums)))} & sidechain & ~H')
    for a in endpoints:
        r=getattr(a,'residue',None)
        if r is None: continue
        mid=next((k for k,v in structures.items() if v is getattr(r,'structure',None)),None)
        if mid and (getattr(a.element,'name','')=='H' or getattr(a,'is_backbone',False)):
            chosen.append(a.atomspec)
    run(session,'select clear')
    if chosen:
        run(session,'select '+chosen[0])
        for x in chosen[1:]: run(session,'select add '+x)
        run(session,'show sel atoms'); run(session,'style sel stick'); run(session,'color sel byhetero')
    # Hide non-endpoint hydrogens on displayed residues.
    for r in residues:
        for a in r.atoms:
            if getattr(a.element,'name','')=='H': a.display = a in endpoints
    run(session,'select clear')

def selected_groups(session,id1,cs1,id2,cs2):
    wanted={(id1,a,id2,b) for a in cs1 for b in cs2}
    # Also accept reverse orientation so show is symmetric.
    reverse={(id2,b,id1,a) for a in cs1 for b in cs2}
    groups=[]
    for m in all_models(session):
        d=group_details(m)
        if d and d[:4] in wanted|reverse: groups.append((m,d))
    return groups

def show_interactions(session,arguments):
    try:
        id1,cs1,id2,cs2,keeps,dist,hide_others=parse(arguments)
        s1=find_structure(session,id1); s2=find_structure(session,id2)
        if s1 is None: raise ValueError('Could not locate coordinate model #'+id1)
        if s2 is None: raise ValueError('Could not locate coordinate model #'+id2)
        structures={id1:s1,id2:s2}; allowed={(id1,c) for c in cs1}|{(id2,c) for c in cs2}
        allm=all_models(session); chosen=selected_groups(session,id1,cs1,id2,cs2)
        if not chosen: raise ValueError('No interaction groups match those model/chain endpoints.')
        # Hide all groups belonging to this endpoint-model pair, then reveal requested pairs.
        relevant=[]
        for m in allm:
            d=group_details(m)
            if d and ({d[0],d[2]}=={id1,id2}): relevant.append((m,d))
        old_labels={id(g):[(c,c.display) for c in label_models(g)] for g,_ in chosen}
        for g,_ in relevant:
            g.display=False
            for pb in g.pseudobonds: pb.display=False
            for c in label_models(g): c.display=False
        if hide_others:
            run(session,f'hide #{id1} atoms'); run(session,f'hide #{id2} atoms')
        else:
            for c in cs1: run(session,f'hide #{id1}/{c} atoms')
            for c in cs2: run(session,f'hide #{id2}/{c} atoms')
        residues=set(); endpoints=set(); count=0
        for g,(m1,c1,m2,c2,kind) in chosen:
            gid=tuple(g.id)
            for parent in allm:
                pid=tuple(getattr(parent,'id',()))
                if pid and len(pid)<len(gid) and gid[:len(pid)]==pid: parent.display=True
            g.display=True
            pair_allowed={(m1,c1),(m2,c2)}
            for pb in g.pseudobonds:
                for a in pb.atoms:
                    r=residue_for_atom(a,structures,pair_allowed)
                    if r is not None: residues.add(r)
                    endpoints.add(a); a.display=True
                    try: a.hide=0
                    except Exception: pass
                    marker=getattr(a,'structure',None)
                    if marker is not None and marker not in structures.values():
                        marker.display=True
                        if kind=='pi': a.radius=0.001
                try: pb.shown_when_atoms_hidden=True
                except Exception: pass
                pb.display=True; count+=1
            for child,previous in old_labels[id(g)]: child.display=previous if dist is None else dist
        for mid,ch,nums in keeps:
            model=structures.get(mid) or find_structure(session,mid)
            if model is None: raise ValueError('Could not locate keep model #'+mid)
            structures[mid]=model
            residues.update(r for r in model.residues if r.chain_id==ch and r.number in nums)
        # Cartoon visibility on each endpoint model.
        for mid,model,selected in ((id1,s1,set(cs1)),(id2,s2,set(cs2))):
            available={r.chain_id for r in model.residues}
            if hide_others:
                for ch in sorted(available): run(session,f'{"show" if ch in selected else "hide"} #{mid}/{ch} cartoons')
            else:
                for ch in sorted(selected & available): run(session,f'show #{mid}/{ch} cartoons')
        show_residues(session,structures,residues,endpoints)
        session.logger.info(f'[show_interactions] #{id1}/{"-".join(cs1)} <-> #{id2}/{"-".join(cs2)}: {count} interaction pseudobonds and {len(residues)} unique residues')
    except ValueError as e:
        session.logger.error('[show_interactions] Error: '+str(e))

def hide_interactions(session,arguments):
    try:
        id1,cs1,id2,cs2,_,_,_=parse(arguments)
        chosen=selected_groups(session,id1,cs1,id2,cs2)
        if not chosen: raise ValueError('No interaction groups match those model/chain endpoints.')
        hidden=0
        for g,_ in chosen:
            g.display=False
            for pb in g.pseudobonds: pb.display=False; hidden+=1
            for c in label_models(g): c.display=False
        for c in cs1: run(session,f'hide #{id1}/{c} atoms')
        for c in cs2: run(session,f'hide #{id2}/{c} atoms')
        session.logger.info(f'[hide_interactions] hid {hidden} interaction pseudobonds')
    except ValueError as e:
        session.logger.error('[hide_interactions] Error: '+str(e))

register('show_interactions',CmdDesc(required=[('arguments',RestOfLine)],synopsis='Show existing multi-model interaction groups and protein residues'),show_interactions,logger=session.logger)
register('hide_interactions',CmdDesc(required=[('arguments',RestOfLine)],synopsis='Hide existing multi-model interaction groups and protein atoms'),hide_interactions,logger=session.logger)
print('[show_interactions] Multi-model commands initialized: show_interactions, hide_interactions')
