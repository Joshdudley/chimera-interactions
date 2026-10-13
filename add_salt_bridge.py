# add_salt_bridge.py
# Register as a true native command inside ChimeraX.
# Usage in command line: add_salt_bridge #4 A B-F printFile yes #4.1 [showLabels yes|no]

import sys
import os
import numpy as np
from chimerax.core.commands import run, register, CmdDesc, StringArg


def _parse_show_labels(value):
    v = str(value).strip().lower()
    if v in ("yes", "true", "on", "1"):
        return True
    if v in ("no", "false", "off", "0"):
        return False
    raise ValueError("showLabels must be yes/no, true/false, on/off, or 1/0")

def _native_add_salt_bridge(session, model_input, chain1_id, chain2_id, print_file_keyword, print_file_value, interaction_model_id, showLabels="yes"):
    # 1. Standardize the user-defined model input as a clean prefix string
    source_model_input = str(model_input).strip()
    if not source_model_input.startswith("#") or source_model_input.startswith("##"):
        session.logger.error(
            f"[add_salt_bridge] Error: source model must include '#', for example #4."
        )
        return
    target_id_str = source_model_input[1:]
    if (not target_id_str
            or any(part == "" or not part.isdigit() for part in target_id_str.split("."))):
        session.logger.error(
            f"[add_salt_bridge] Error: source model must be a numeric model ID such as #4 or #4.2."
        )
        return

    # 2. Parse the user-defined secondary destination chain boundaries (e.g., 'B-F' -> ['B','C','D','E','F'])
    target_chains2 = []
    if "-" in chain2_id:
        start_c, end_c = chain2_id.split("-")
        for ord_c in range(ord(start_c), ord(end_c) + 1):
            target_chains2.append(chr(ord_c))
    else:
        target_chains2 = [chain2_id]

    # 3. Locate your specific target structure using a robust string-prefix match
    struct_model = None
    # Require the exact source model requested by the user.
    for m in session.models:
        if hasattr(m, 'atoms'):
            m_id_str = ".".join(map(str, m.id))
            if m_id_str == target_id_str:
                struct_model = m
                break

    if not struct_model:
        session.logger.error(f"[add_salt_bridge] Error: Could not locate molecular coordinate model matching {source_model_input}")
        return

    model_spec = f"#{'.'.join(map(str, struct_model.id))}"
    print(f"[add_salt_bridge] Actively analyzing molecular coordinate structure model: {model_spec}")

    # `printFile yes 4.1` is intentionally parsed as three positional
    # arguments. This preserves the exact user-facing syntax without making
    # ChimeraX interpret the trailing interaction model ID as an illegal
    # positional argument after a keyword.
    if str(print_file_keyword).strip().lower() not in ("printf_file", "print_file", "printfile"):
        session.logger.error(
            "Usage: add_salt_bridge #MODEL CHAIN1 CHAIN2 printFile yes #INTERACTION_MODEL_ID"
        )
        return
    print_file_mode = str(print_file_value).strip().lower()
    if print_file_mode not in ("yes", "no", "true", "false", "on", "off", "1", "0"):
        session.logger.error("Error: printFile must be yes/no, true/false, on/off, or 1/0.")
        return
    write_file = print_file_mode in ("yes", "true", "on", "1")

    interaction_input = str(interaction_model_id).strip()
    if not interaction_input.startswith("#") or interaction_input.startswith("##"):
        session.logger.error(
            f"[add_salt_bridge] Error: destination model must include '#', for example #4.1."
        )
        return

    interaction_id_str = interaction_input[1:]
    if (not interaction_id_str
            or any(part == "" or not part.isdigit() for part in interaction_id_str.split("."))):
        session.logger.error(
            f"[add_salt_bridge] Error: destination model must be a numeric hierarchical ID such as #4.1 or #4.7."
        )
        return

    try:
        show_labels = _parse_show_labels(showLabels)
    except ValueError as e:
        session.logger.error(f"[add_salt_bridge] Error: {e}")
        return

    nested_model_spec = interaction_input
    source_name_id = target_id_str.replace(".", "_")
    destination_name_id = interaction_id_str.replace(".", "_")
    # ChimeraX tracks pseudobond groups by name. Use the final, unique
    # chain-pair name when creating the group, including empty groups.
    final_group_name = f"Model_{source_name_id}_salt_bridge_Chain_{chain1_id}_to_Chain_{chain2_id}"
    temporary_group_name = final_group_name

    # Prevent duplicate source-derived group names from colliding.
    for m in session.models:
        if (hasattr(m, 'pseudobonds') and not hasattr(m, 'atoms')
                and getattr(m, 'name', '') == final_group_name):
            session.logger.error(
                f"[add_salt_bridge] Error: interaction group '{final_group_name}' already exists. "
                "Remove the existing group before creating another one for this source model."
            )
            return

    # Salt bridges are restricted to oppositely charged sidechains:
    # ASP/GLU carboxylates versus LYS/ARG cationic nitrogens.
    acidic = {
        "ASP": {"OD1", "OD2"},
        "GLU": {"OE1", "OE2"},
    }
    basic = {
        "LYS": {"NZ"},
        "ARG": {"NE", "NH1", "NH2"},
    }

    acid_c1 = [a for a in struct_model.atoms if a.residue.chain_id == chain1_id and a.name in acidic.get(a.residue.name, set())]
    base_c1 = [a for a in struct_model.atoms if a.residue.chain_id == chain1_id and a.name in basic.get(a.residue.name, set())]
    acid_c2 = [a for a in struct_model.atoms if a.residue.chain_id in target_chains2 and a.name in acidic.get(a.residue.name, set())]
    base_c2 = [a for a in struct_model.atoms if a.residue.chain_id in target_chains2 and a.name in basic.get(a.residue.name, set())]

    print(f"[add_salt_bridge] Found {len(acid_c1)} acidic + {len(base_c1)} basic atoms on Chain {chain1_id} and {len(acid_c2)} acidic + {len(base_c2)} basic atoms on Chains {chain2_id}")

    residue_pairs_map = {}
    affected_residues = set()

    def consider(a1, a2):
        if a1.residue == a2.residue:
            return
        res_pair_key = (a1.residue, a2.residue)
        d = float(np.linalg.norm(a1.coord - a2.coord))
        if d > 4.0:
            return
        if res_pair_key not in residue_pairs_map or d < residue_pairs_map[res_pair_key]["dist"]:
            residue_pairs_map[res_pair_key] = {
                "atom1": a1, "atom2": a2, "dist": d,
                "res1": a1.residue, "res2": a2.residue
            }

    for a1 in acid_c1:
        for b2 in base_c2:
            consider(a1, b2)
    for b1 in base_c1:
        for a2 in acid_c2:
            consider(b1, a2)

    # 6. Initialize trackers for metrics auditing
    pbond_count = 0
    chain_counts = {chain1_id: 0}
    for ch in target_chains2:
        chain_counts[ch] = 0

    pbond_lines_output = []

    # 7. Run un-nested global group call to generate the native toggleable labels subitem flawlessly
    for data in residue_pairs_map.values():
        at1 = data['atom1']
        at2 = data['atom2']
        affected_residues.add(data['res1'])
        affected_residues.add(data['res2'])
        
        # Update chain counters
        chain_counts[at1.residue.chain_id] += 1
        chain_counts[at2.residue.chain_id] += 1
        
        # Format the specific atom specifier tokens
        spec1 = f"{model_spec}/{at1.residue.chain_id}:{at1.residue.number}@{at1.name}"
        spec2 = f"{model_spec}/{at2.residue.chain_id}:{at2.residue.number}@{at2.name}"
        
        # Visualize the actual charged atoms selected by the salt-bridge chemistry.
        pbond_cmd = f'pbond {spec1} {spec2} color orange radius 0.2 dashes 8 showDist true name "{temporary_group_name}" global'
        run(session, pbond_cmd)
        
        pbond_file_line = f"pbond {spec1} {spec2} color orange radius 0.2 dashes 8 showDist true name 'salt_bridge' global"
        print(pbond_file_line)
        pbond_lines_output.append(pbond_file_line)
        pbond_count += 1

    # Discover generated group model ID safely via name match lookup
    # Find the exact pseudobond group created by this command invocation.
    detected_id_str = None
    for m in session.models:
        if (hasattr(m, 'pseudobonds') and not hasattr(m, 'atoms')
                and getattr(m, 'name', '') == temporary_group_name):
            detected_id_str = f"#{'.'.join(map(str, m.id))}"
            break

    if not detected_id_str:
        # A valid interface may contain no contacts. Keep its empty subgroup.
        try:
            from chimerax.atomic import PseudobondManager
            empty_group = PseudobondManager(session).get_group(temporary_group_name, create=True)
            if empty_group is None:
                raise RuntimeError("PseudobondManager.get_group() returned None")
            if not any(m is empty_group for m in _suite_all_models(session)):
                session.models.add([empty_group])
            detected_id_str = _suite_model_spec(empty_group)
            session.logger.info(f"[add_salt_bridge] No contacts found for {nested_model_spec}; created an empty subgroup.")
        except Exception as e:
            session.logger.error(f"[add_salt_bridge] ERROR: Could not create empty subgroup for {nested_model_spec}: {e}")
            return

    # Assign only the user-requested hierarchical ID. Keep the generated
    # pseudobond-group name unique so existing groups cannot cause a name collision.
    rename_cmd = f'rename {detected_id_str} id {interaction_id_str}'
    run(session, rename_cmd)

    assigned_model = None
    for m in session.models:
        if ("#" + ".".join(map(str, m.id)) == nested_model_spec
                and hasattr(m, 'pseudobonds') and not hasattr(m, 'atoms')):
            assigned_model = m
            break

    if assigned_model is None:
        session.logger.error(
            f"[add_salt_bridge] ERROR: ChimeraX did not assign the generated interaction group to {nested_model_spec}. "
            "The requested model ID may already be occupied."
        )
        return

    session.logger.info(
        f"[add_salt_bridge] Successfully assigned interaction model {nested_model_spec} "
        f"with group name '{final_group_name}'"
    )

    # Always create the ChimeraX distance-label child model via showDist true.
    # showLabels only controls its initial visibility. When false, keep the child
    # object in the hierarchy but hide that child so it can be re-enabled later.
    if not show_labels:
        # IMPORTANT: showDist must remain TRUE above so ChimeraX creates the
        # distance-label object.  We only hide the existing child model here.
        # Use the assigned interaction model's full hierarchy rather than
        # session.models, because child models are not reliably returned by
        # the latter in all ChimeraX versions.
        try:
            children = list(assigned_model.all_models())[1:]
        except Exception:
            children = []
        for child in children:
            child_name = str(getattr(child, "name", "")).strip().lower()
            if child_name in {"distance", "distances", "labels"}:
                try:
                    child.display = False
                except Exception:
                    try:
                        run(session, f"hide #{'.'.join(map(str, child.id))} models")
                    except Exception:
                        pass

    if write_file:
        output_filename = "salt_bridge.txt"
        try:
            with open(output_filename, "w") as f:
                f.write("\n".join(pbond_lines_output) + "\n\n")
                f.write("=" * 50 + "\n")
                f.write("      INTERFACE INTERACTION METRICS SUMMARY      \n")
                f.write("=" * 50 + "\n")
                f.write(f"Total number of salt bridges found: {pbond_count}\n")
                f.write("-" * 50 + "\n")
                for chain_name, count_val in sorted(chain_counts.items()):
                    f.write(f"  • Chain {chain_name} : {count_val} salt bridge contacts found.\n")
                f.write("=" * 50 + "\n")
            print(f"[add_salt_bridge] Successfully exported text mapping table to: {os.path.abspath(output_filename)}")
        except Exception as e:
            print(f"Error writing file: {e}")

    # Visual formatting optimization loop block passes
    if affected_residues:
        for res in affected_residues:
            for atom in res.atoms:
                if not atom.is_backbone:
                    atom.display = True
                    atom.style = 1
                    atom.radius = 0.3
            if hasattr(res.atoms, 'bonds') and res.atoms.bonds:
                for bond in res.atoms.bonds:
                    b_atoms = bond.atoms
                    # Fixed tuple unpacking index references matching hydrophobic script specs
                    if len(b_atoms) == 2 and not b_atoms[0].is_backbone and not b_atoms[1].is_backbone:
                        bond.radius = 0.2  # Enforce uniform stick thickness
                    
        run(session, f"color {model_spec} & sidechain byhetero")
        
    run(session, "select clear")
    
    print("\n" + "="*50)
    print("      INTERFACE INTERACTION METRICS PER CHAIN      ")
    print("="*50)
    for chain_name, count_val in sorted(chain_counts.items()):
        print(f"  • Chain {chain_name} : {count_val} salt bridge contacts found.")
    print("="*50)
    print(f"Successfully generated exactly {pbond_count} single-line interactions inside group '{final_group_name}' at model '{nested_model_spec}'!\n")

# Complete correct registration hooks matching naming schema definitions


# ---------------------------------------------------------------------------
# Chain-pair hierarchy wrapper added without changing the native detector body.
# The original detector remains the chemistry authority and is run once per
# target chain. Each successful result is moved under a family parent.
# ---------------------------------------------------------------------------
from chimerax.core.models import Model


def _suite_all_models(session):
    seen=set(); out=[]
    def walk(items):
        for m in items:
            if id(m) in seen:
                continue
            seen.add(id(m)); out.append(m)
            try:
                children=list(m.child_models())
            except Exception:
                children=[]
            walk(children)
    walk(list(session.models))
    return out


def _suite_model_spec(model):
    return '#' + '.'.join(map(str, model.id))


def _suite_find_model(session, spec):
    s=str(spec).strip()
    if not s.startswith('#'):
        return None
    try:
        wanted=tuple(int(x) for x in s[1:].split('.'))
    except Exception:
        return None
    for m in _suite_all_models(session):
        if tuple(getattr(m,'id',())) == wanted:
            return m
    return None


def _suite_chain_list(spec):
    s=str(spec).strip()
    if '-' not in s:
        return [s] if s else []
    parts=s.split('-')
    if len(parts)!=2 or len(parts[0])!=1 or len(parts[1])!=1:
        raise ValueError(f'Chain range must be like B-F, got {s}')
    a,b=parts
    step=1 if ord(b)>=ord(a) else -1
    return [chr(c) for c in range(ord(a),ord(b)+step,step)]


def _suite_hide_group(group):
    try: group.display=False
    except Exception: pass
    for pb in getattr(group,'pseudobonds',()):
        try: pb.display=False
        except Exception: pass
    try: children=list(group.all_models())[1:]
    except Exception: children=[]
    for child in children:
        name=str(getattr(child,'name','')).lower()
        if 'distance' in name or 'label' in name:
            try: child.display=False
            except Exception: pass


def _suite_snapshot(source):
    atoms=[(a,getattr(a,'display',False),getattr(a,'style',None),getattr(a,'radius',None)) for a in source.atoms]
    bonds=[]; seen=set()
    for a in source.atoms:
        for b in getattr(a,'bonds',()):
            if id(b) in seen: continue
            seen.add(id(b)); bonds.append((b,getattr(b,'display',False),getattr(b,'radius',None)))
    return atoms,bonds


def _suite_restore(snap):
    atoms,bonds=snap
    for a,d,style,radius in atoms:
        try: a.display=d
        except Exception: pass
        try: a.style=style
        except Exception: pass
        try: a.radius=radius
        except Exception: pass
    for b,d,radius in bonds:
        try: b.display=d
        except Exception: pass
        try: b.radius=radius
        except Exception: pass


def add_salt_bridge(session, model_input, chain1_id, chain2_id, print_file_keyword, print_file_value, interaction_model_id, showLabels='yes'):
    chain2_list=_suite_chain_list(chain2_id)
    if not chain2_list:
        session.logger.error('[add_salt_bridge] No target chains were supplied.')
        return
    source=_suite_find_model(session,model_input)
    if source is None or not hasattr(source,'atoms'):
        session.logger.error(f'[add_salt_bridge] Could not locate coordinate model {model_input}.')
        return
    parent_spec=str(interaction_model_id).strip()
    if not parent_spec.startswith('#'):
        session.logger.error('[add_salt_bridge] Parent interaction model must look like #2.1.')
        return
    if _suite_find_model(session,parent_spec) is not None:
        session.logger.error(f'[add_salt_bridge] Parent model {parent_spec} already exists. Remove it before rebuilding this interaction family.')
        return

    source_id='_'.join(map(str,source.id))
    parent=Model(f'Model_{source_id}_salt_bridge',session)
    session.models.add([parent])
    run(session, f'rename {_suite_model_spec(parent)} name "Model_{source_id}_salt_bridge" id {parent_spec}')
    parent=_suite_find_model(session,parent_spec)
    if parent is None:
        session.logger.error(f'[add_salt_bridge] Could not create parent interaction model {parent_spec}.')
        return

    counts={}; created=[]
    for idx,c2 in enumerate(chain2_list,start=1):
        child_spec=parent_spec+'.'+str(idx)
        snap=_suite_snapshot(source)
        before_ids={id(m) for m in _suite_all_models(session)}
        try:
            _native_add_salt_bridge(session,_suite_model_spec(source),str(chain1_id),str(c2),'printFile','no',child_spec,showLabels)
        finally:
            _suite_restore(snap)

        child=_suite_find_model(session,child_spec)
        if child is None or not hasattr(child,'pseudobonds'):
            counts[c2]=0
            continue

        # The native detector created this subgroup with its final chain-pair name.
        try:
            child._interaction_suite_kind='salt_bridge'
            child._interaction_suite_source_id=source_id
            child._interaction_suite_chain1=str(chain1_id)
            child._interaction_suite_chain2=str(c2)
        except Exception:
            pass
        _suite_hide_group(child)
        created.append(child)
        counts[c2]=len(list(getattr(child,'pseudobonds',())))

        if 'salt_bridge'=='pi':
            for m in _suite_all_models(session):
                if str(getattr(m,'name','')).startswith('pi_centroids_') and id(m) not in before_ids:
                    try: m.display=False
                    except Exception: pass

    mode=str(print_file_value).strip().lower()
    total=sum(counts.values())
    print('\n'+'='*50)
    print('      INTERFACE INTERACTION METRICS PER CHAIN      ')
    print('='*50)
    for c2 in chain2_list:
        print(f'  - Chain {c2} : {counts.get(c2,0)} salt bridge interaction contacts found.')
    print('='*50)
    print(f'Created parent {parent_spec} with {len(created)} chain-pair subgroup(s); total contacts: {total}')

    if mode in ('yes','true','on','1'):
        filename='salt_bridge.txt'
        with open(filename,'w') as f:
            f.write('='*50+'\n')
            f.write('      INTERFACE INTERACTION METRICS SUMMARY      \n')
            f.write('='*50+'\n')
            f.write('Total number of salt bridge interactions found: '+str(total)+'\n')
            f.write('-'*50+'\n')
            for c2 in chain2_list:
                f.write(f'  - Chain {c2} : {counts.get(c2,0)} salt bridge interaction contacts found.\n')
            f.write('='*50+'\n')


desc = CmdDesc(
    required=[("model_input", StringArg), ("chain1_id", StringArg), ("chain2_id", StringArg), ("print_file_keyword", StringArg), ("print_file_value", StringArg), ("interaction_model_id", StringArg)],
    keyword=[("showLabels", StringArg)],
    synopsis="Generate salt bridge interaction subgroups by chain pair"
)
register("add_salt_bridge", desc, add_salt_bridge, logger=session.logger)
print("[add_salt_bridge] Command initialized successfully with chain-pair subgroups.")
