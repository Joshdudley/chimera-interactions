# add_pi.py
# Register as a true native command inside ChimeraX.
# Usage in command line: add_pi #4 A B-F printFile yes #4.1

import os
import sys
import numpy as np
from chimerax.core.commands import run, register, CmdDesc, StringArg


def _parse_show_labels(value):
    """Parse the optional showLabels keyword. Default behavior is to show distance labels."""
    v = str(value).strip().lower()
    if v in ("yes", "true", "on", "1"):
        return True
    if v in ("no", "false", "off", "0"):
        return False
    raise ValueError("showLabels must be yes/no, true/false, on/off, or 1/0")
from chimerax.markers import MarkerSet

PI_PI_CUTOFF = 6.5
PARTNER_CUTOFF = 5.5

def _native_add_pi(session, model_input, chain1_id, chain2_id, print_file_keyword, print_file_value, interaction_model_id, showLabels="yes"):
    # 1. Define specific chemical residue groupings for aromatic networks
    aromatic_res = ["PHE", "TYR", "TRP", "HIS"]
    partner_res  = ["ARG", "LYS", "MET", "CYS", "PHE", "TYR", "TRP", "HIS"]

    source_model_input = str(model_input).strip()
    if not source_model_input.startswith("#") or source_model_input.startswith("##"):
        session.logger.error(
            f"[add_pi] Error: source model must include '#', for example #4."
        )
        return
    target_id_str = source_model_input[1:]
    if (not target_id_str
            or any(part == "" or not part.isdigit() for part in target_id_str.split("."))):
        session.logger.error(
            f"[add_pi] Error: source model must be a numeric model ID such as #4 or #4.2."
        )
        return

    # 2. Parse secondary destination chain boundaries safely
    target_chains2 = []
    if "-" in chain2_id:
        start_c, end_c = chain2_id.split("-")
        for ord_c in range(ord(start_c), ord(end_c) + 1):
            target_chains2.append(chr(ord_c))
    else:
        target_chains2 = [chain2_id]

    # 3. Locate target coordinate structure
    struct_model = None
    # Require the exact source model requested by the user.
    for m in session.models:
        if hasattr(m, 'atoms'):
            m_id_str = ".".join(map(str, m.id))
            if m_id_str == target_id_str:
                struct_model = m
                break

    if not struct_model:
        session.logger.error(f"[add_pi] Error: Could not locate molecular coordinate model matching {source_model_input}")
        return

    model_spec = f"#{'.'.join(map(str, struct_model.id))}"
    session.logger.info(f"[add_pi] Actively analyzing molecular coordinate structure model: {model_spec}")

    # `printFile yes 4.1` is intentionally parsed as three positional
    # arguments. This preserves the exact user-facing syntax without making
    # ChimeraX interpret the trailing interaction model ID as an illegal
    # positional argument after a keyword.
    if str(print_file_keyword).strip().lower() not in ("printf_file", "print_file", "printfile"):
        session.logger.error(
            "Usage: add_pi #MODEL CHAIN1 CHAIN2 printFile yes #INTERACTION_MODEL_ID [showLabels yes|no]"
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
            f"[add_pi] Error: destination model must include '#', for example #4.1."
        )
        return

    interaction_id_str = interaction_input[1:]
    if (not interaction_id_str
            or any(part == "" or not part.isdigit() for part in interaction_id_str.split("."))):
        session.logger.error(
            f"[add_pi] Error: destination model must be a numeric hierarchical ID such as #4.1 or #4.7."
        )
        return

    try:
        show_labels = _parse_show_labels(showLabels)
    except ValueError as e:
        session.logger.error(f"[add_pi] Error: {e}")
        return

    nested_model_spec = interaction_input
    source_name_id = target_id_str.replace(".", "_")
    destination_name_id = interaction_id_str.replace(".", "_")
    # Pseudobond group names are keys in ChimeraX's pseudobond manager. Create
    # each group with its final chain-pair name; changing that key after the
    # first group is created can leave later groups with temporary names.
    final_group_name = f"Model_{source_name_id}_pi_Chain_{chain1_id}_to_Chain_{chain2_id}"
    temporary_group_name = final_group_name

    # Prevent duplicate source-derived group names from colliding.
    for m in session.models:
        if (hasattr(m, 'pseudobonds') and not hasattr(m, 'atoms')
                and getattr(m, 'name', '') == final_group_name):
            session.logger.error(
                f"[add_pi] Error: interaction group '{final_group_name}' already exists. "
                "Remove the existing group before creating another one for this source model."
            )
            return

    # Discover residues first, then inspect their side-chain atoms.  This avoids
    # relying on ChimeraX atom-level classification to decide whether an entire
    # residue is eligible.  The user-facing chain syntax remains exactly as
    # requested (e.g. B-F); only chains actually present in the model contribute.
    residue_by_side = {}
    for atom in struct_model.atoms:
        residue = getattr(atom, "residue", None)
        if residue is None:
            continue
        chain = str(getattr(residue, "chain_id", "")).strip().upper()
        resname = str(getattr(residue, "name", "")).strip().upper()
        if chain not in [str(chain1_id).strip().upper()] + [str(c).strip().upper() for c in target_chains2]:
            continue
        if resname not in partner_res:
            continue
        residue_by_side[(chain, residue.number, resname)] = residue

    residues_c1_pre = sorted(
        [r for (chain, _, _), r in residue_by_side.items() if chain == str(chain1_id).strip().upper()],
        key=lambda r: (r.number, r.name)
    )
    residues_c2_pre = sorted(
        [r for (chain, _, _), r in residue_by_side.items() if chain in {str(c).strip().upper() for c in target_chains2}],
        key=lambda r: (str(r.chain_id), r.number, r.name)
    )

    atoms_c1 = [a for r in residues_c1_pre for a in getattr(r, "atoms", [])]
    atoms_c2 = [a for r in residues_c2_pre for a in getattr(r, "atoms", [])]

    present_target_chains = sorted({str(r.chain_id).strip().upper() for r in residues_c2_pre})
    print(f"[add_pi] Requested comparison: Chain {chain1_id} vs {chain2_id} (present target chains: {present_target_chains or 'none'})")
    print(f"[add_pi] Eligible residues on Chain {chain1_id}: {len(residues_c1_pre)}")
    print(f"[add_pi] Eligible residues on Chains {chain2_id}: {len(residues_c2_pre)}")
    print(f"[add_pi] Candidate atoms on Chain {chain1_id}: {len(atoms_c1)}; on Chains {chain2_id}: {len(atoms_c2)}")

    ring_names = {
        "PHE": {"CG", "CD1", "CD2", "CE1", "CE2", "CZ"},
        "TYR": {"CG", "CD1", "CD2", "CE1", "CE2", "CZ"},
        "TRP": {"CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2"},
        "HIS": {"CG", "ND1", "CD2", "CE1", "NE2"},
    }

    # Chemistry-specific partner definitions. We intentionally do not require
    # ring-plane orientation: this is a permissive snapshot detector intended to
    # capture plausible interactions that may be strengthened by conformational
    # breathing in nearby structures/frames.
    cation_atoms = {
        "LYS": {"NZ"},
        "ARG": {"NE", "NH1", "NH2"},
    }
    sulfur_atoms = {
        "MET": {"SD"},
        "CYS": {"SG"},
    }

    def ring_atoms_for_residue(residue):
        names = ring_names.get(residue.name, set())
        return [a for a in residue.atoms if a.name in names]

    def atoms_named(residue, allowed_names):
        return [a for a in residue.atoms if a.name in allowed_names]

    def centroid(atoms):
        return np.mean(np.array([a.coord for a in atoms]), axis=0)

    def candidate_for_pair(res1, res2):
        """Return (atom1, atom2, distance, subtype) or None."""
        r1_ring = ring_atoms_for_residue(res1)
        r2_ring = ring_atoms_for_residue(res2)

        # Aromatic-aromatic: use ring-centroid distance for detection and
        # store centroid coordinates for the true centroid-to-centroid display.
        if r1_ring and r2_ring:
            c1 = centroid(r1_ring)
            c2 = centroid(r2_ring)
            dist = float(np.linalg.norm(c1 - c2))
            nearest_atom_dist = min(float(np.linalg.norm(a.coord - b.coord)) for a in r1_ring for b in r2_ring)
            if dist <= PI_PI_CUTOFF or nearest_atom_dist <= PARTNER_CUTOFF:
                return None  # handled in dedicated rendering path below
            return None

        # Exactly one side is aromatic. Identify the non-aromatic partner by
        # chemistry rather than by an arbitrary nearest side-chain atom.
        if r1_ring:
            aromatic_res, partner_res = res1, res2
            aromatic_atoms = r1_ring
            partner_side = 2
        elif r2_ring:
            aromatic_res, partner_res = res2, res1
            aromatic_atoms = r2_ring
            partner_side = 1
        else:
            return None

        ring_center = centroid(aromatic_atoms)

        if partner_res.name in cation_atoms:
            candidates = atoms_named(partner_res, cation_atoms[partner_res.name])
            subtype = "cation-pi"
        elif partner_res.name in sulfur_atoms:
            candidates = atoms_named(partner_res, sulfur_atoms[partner_res.name])
            subtype = "sulfur-pi"
        else:
            return None

        if not candidates:
            return None

        partner_atom = min(candidates, key=lambda a: np.linalg.norm(ring_center - a.coord))
        dist = float(np.linalg.norm(ring_center - partner_atom.coord))
        if dist > PARTNER_CUTOFF:
            return None

        if partner_side == 2:
            return aromatic_res, partner_atom, dist, subtype
        return partner_atom, aromatic_res, dist, subtype

    # Create a dedicated marker set for true ring-centroid endpoints used in
    # pi-pi and aromatic-partner visualizations. The marker set must exist before
    # the pseudobonds are created because the pseudobonds use the marker atoms as
    # endpoints. It is kept as a separate top-level model to avoid ChimeraX model
    # DAG cycles. We deliberately move it away from the user-requested interaction
    # ID so that, for example, #7 remains available for the pi interaction group.
    marker_set = MarkerSet(session, name=f"pi_centroids_{target_id_str}_{destination_name_id}")
    session.models.add([marker_set])

    # Reserve the requested interaction ID by relocating the marker set to a
    # high, unused top-level numeric model ID before any interaction group is
    # assigned. This prevents the marker set from occupying #7 when the caller
    # requests add_pi ... #7.
    try:
        top_level_ids = []
        for existing in session.models:
            eid = tuple(getattr(existing, "id", ()))
            if len(eid) == 1 and isinstance(eid[0], int):
                top_level_ids.append(eid[0])
        safe_marker_id = max(top_level_ids + [1000]) + 1
        marker_id_now = f"#{marker_set.id[0]}"
        if marker_set.id[0] != safe_marker_id:
            run(session, f"rename {marker_id_now} id {safe_marker_id}")
    except Exception as e:
        session.logger.warning(f"[add_pi] Could not relocate centroid marker model before interaction-group assignment: {e}")
    marker_atoms = {}
    # Persist an exact marker -> source-residue mapping on the MarkerSet itself.
    # This lets downstream tools such as show_interactions read the existing
    # add_pi pseudobonds without re-detecting pi chemistry or guessing from
    # centroid coordinates.
    marker_set._pi_source_residue_map = {}

    def centroid_marker(residue, ring_atoms):
        key = (residue.chain_id, residue.number, residue.name)
        if key not in marker_atoms:
            m = marker_set.create_marker(centroid(ring_atoms), (255,165,0,255), 0.05, id=len(marker_atoms)+1)
            marker_atoms[key] = m
            marker_set._pi_source_residue_map[id(m)] = residue
        return marker_atoms[key]

    def explicit_hydrogens_on_atom(atom):
        hs = []
        try:
            for bond in atom.bonds:
                for other in bond.atoms:
                    if other is not atom and getattr(other.element, 'name', '') == 'H':
                        hs.append(other)
        except Exception:
            pass
        return hs

    # Search both directions. Each residue pair is stored using a chain/residue
    # order-independent key so aromatic-aromatic pairs cannot be duplicated.
    residues_c1 = residues_c1_pre
    residues_c2 = residues_c2_pre

    residue_pairs_map = {}
    affected_residues = set()
    display_hydrogens = set()

    for res1 in residues_c1:
        for res2 in residues_c2:
            if res1 == res2:
                continue
            r1_ring = ring_atoms_for_residue(res1)
            r2_ring = ring_atoms_for_residue(res2)

            if r1_ring and r2_ring:
                c1 = centroid(r1_ring)
                c2 = centroid(r2_ring)
                dist = float(np.linalg.norm(c1 - c2))
                nearest_atom_dist = min(float(np.linalg.norm(a.coord - b.coord)) for a in r1_ring for b in r2_ring)
                if dist <= PI_PI_CUTOFF or nearest_atom_dist <= PARTNER_CUTOFF:
                    key1 = (res1.chain_id, res1.number, res1.name)
                    key2 = (res2.chain_id, res2.number, res2.name)
                    res_pair_key = tuple(sorted((key1, key2)))
                    residue_pairs_map[res_pair_key] = {
                        'type': 'pi-pi', 'res1': res1, 'res2': res2,
                        'dist': dist, 'centroid1': c1, 'centroid2': c2
                    }
                continue

            if r1_ring or r2_ring:
                aromatic_res = res1 if r1_ring else res2
                partner_residue = res2 if r1_ring else res1
                aromatic_atoms = r1_ring if r1_ring else r2_ring
                center = centroid(aromatic_atoms)
                partner_atom = None
                subtype = None
                partner_display_atom = None

                if partner_residue.name in cation_atoms:
                    cands = atoms_named(partner_residue, cation_atoms[partner_residue.name])
                    if cands:
                        # Prefer the closest explicit H attached to the cationic group;
                        # otherwise use the closest cationic heavy atom as a fallback.
                        h_cands = [h for a in cands for h in explicit_hydrogens_on_atom(a)]
                        if h_cands:
                            partner_display_atom = min(h_cands, key=lambda h: np.linalg.norm(center-h.coord))
                            partner_atom = partner_display_atom
                        else:
                            partner_atom = min(cands, key=lambda a: np.linalg.norm(center-a.coord))
                            partner_display_atom = partner_atom
                        subtype = 'cation-pi'
                elif partner_residue.name in sulfur_atoms:
                    cands = atoms_named(partner_residue, sulfur_atoms[partner_residue.name])
                    if cands:
                        partner_atom = min(cands, key=lambda a: np.linalg.norm(center-a.coord))
                        partner_display_atom = partner_atom
                        subtype = 'sulfur-pi'

                if partner_atom is not None:
                    dist = float(np.linalg.norm(center - partner_atom.coord))
                    if dist <= PARTNER_CUTOFF:
                        key1 = (res1.chain_id, res1.number, res1.name)
                        key2 = (res2.chain_id, res2.number, res2.name)
                        res_pair_key = tuple(sorted((key1, key2)))
                        residue_pairs_map[res_pair_key] = {
                            'type': subtype, 'res1': res1, 'res2': res2,
                            'dist': dist, 'partner_atom': partner_display_atom,
                            'aromatic_res': aromatic_res, 'aromatic_atoms': aromatic_atoms
                        }
                        if partner_display_atom.element.name == 'H':
                            display_hydrogens.add(partner_display_atom)

    # Diagnostics: retain the closest plausible candidate for each subtype even
    # when it falls outside the acceptance cutoff. This makes a zero-hit result
    # interpretable for single snapshots of breathing proteins.
    closest_candidates = {"pi-pi": None, "cation-pi": None, "sulfur-pi": None}
    for res1 in residues_c1:
        for res2 in residues_c2:
            if res1 == res2:
                continue
            r1_ring = ring_atoms_for_residue(res1)
            r2_ring = ring_atoms_for_residue(res2)
            if r1_ring and r2_ring:
                c1 = centroid(r1_ring); c2 = centroid(r2_ring)
                d = float(np.linalg.norm(c1 - c2))
                nearest = min(float(np.linalg.norm(a.coord - b.coord)) for a in r1_ring for b in r2_ring)
                score_d = min(d, nearest)
                if closest_candidates["pi-pi"] is None or d < closest_candidates["pi-pi"][0]:
                    closest_candidates["pi-pi"] = (score_d, res1, res2)
            elif r1_ring or r2_ring:
                aromatic_res = res1 if r1_ring else res2
                partner_residue = res2 if r1_ring else res1
                center = centroid(r1_ring if r1_ring else r2_ring)
                if partner_residue.name in cation_atoms:
                    candidates = atoms_named(partner_residue, cation_atoms[partner_residue.name])
                    subtype = "cation-pi"
                elif partner_residue.name in sulfur_atoms:
                    candidates = atoms_named(partner_residue, sulfur_atoms[partner_residue.name])
                    subtype = "sulfur-pi"
                else:
                    candidates = []
                    subtype = None
                if candidates and subtype:
                    d = min(float(np.linalg.norm(center - a.coord)) for a in candidates)
                    if closest_candidates[subtype] is None or d < closest_candidates[subtype][0]:
                        closest_candidates[subtype] = (d, res1, res2)

    # 6. Initialize trackers for metrics auditing
    pbond_count = 0
    chain_counts = {chain1_id: 0}
    for ch in target_chains2:
        chain_counts[ch] = 0

    pbond_lines_output = []

    # 7. Run un-nested global group call to generate the native toggleable labels subitem flawlessly
    for data in residue_pairs_map.values():
        res1 = data['res1']; res2 = data['res2']
        affected_residues.add(res1); affected_residues.add(res2)
        subtype = data['type']

        if subtype == 'pi-pi':
            m1 = centroid_marker(res1, ring_atoms_for_residue(res1))
            m2 = centroid_marker(res2, ring_atoms_for_residue(res2))
            spec1 = f"#{marker_set.id[0]}/M:{m1.residue.number}"
            spec2 = f"#{marker_set.id[0]}/M:{m2.residue.number}"
            chain_counts[res1.chain_id] += 1; chain_counts[res2.chain_id] += 1
            pbond_cmd = f'pbond {spec1} {spec2} color tan radius 0.2 dashes 8 showDist true name "{temporary_group_name}" global'
        else:
            aromatic_res = data['aromatic_res']
            aromatic_atoms = data['aromatic_atoms']
            m = centroid_marker(aromatic_res, aromatic_atoms)
            partner_atom = data['partner_atom']
            centroid_spec = f"#{marker_set.id[0]}/M:{m.residue.number}"
            partner_spec = f"{model_spec}/{partner_atom.residue.chain_id}:{partner_atom.residue.number}@{partner_atom.name}"
            if partner_atom.residue.chain_id == res1.chain_id:
                spec1, spec2 = partner_spec, centroid_spec
            else:
                spec1, spec2 = centroid_spec, partner_spec
            chain_counts[res1.chain_id] += 1; chain_counts[res2.chain_id] += 1
            pbond_cmd = f'pbond {spec1} {spec2} color tan radius 0.2 dashes 8 showDist true name "{temporary_group_name}" global'

        run(session, pbond_cmd)
        if subtype == 'pi-pi':
            pbond_file_line = (f"# {subtype}: {res1.chain_id}:{res1.number} {res1.name} centroid "
                               f"<-> {res2.chain_id}:{res2.number} {res2.name} centroid; "
                               f"centroid distance={data['dist']:.2f} A")
        else:
            aromatic_res = data['aromatic_res']
            partner_atom = data['partner_atom']
            endpoint_kind = 'explicit H' if partner_atom.element.name == 'H' else 'cation/sulfur atom'
            pbond_file_line = (f"# {subtype}: {aromatic_res.chain_id}:{aromatic_res.number} {aromatic_res.name} centroid "
                               f"<-> {partner_atom.residue.chain_id}:{partner_atom.residue.number}@{partner_atom.name} ({endpoint_kind}); "
                               f"distance={data['dist']:.2f} A")
        print(pbond_file_line)
        pbond_lines_output.append(pbond_file_line)
        pbond_count += 1

    # Discover generated parent model ID safely via name match lookup
    # Find the exact pseudobond group created by this command invocation.
    detected_id_str = None
    for m in session.models:
        if (hasattr(m, 'pseudobonds') and not hasattr(m, 'atoms')
                and getattr(m, 'name', '') == temporary_group_name):
            detected_id_str = f"#{'.'.join(map(str, m.id))}"
            break

    if not detected_id_str:
        # Zero-interaction interfaces are valid. Create a real empty
        # PseudobondGroup so the requested chain-pair subgroup still exists.
        # The group is discovered through the model tree after creation rather
        # than assuming get_group() has already assigned a model ID.
        try:
            from chimerax.atomic import PseudobondManager
            empty_group = PseudobondManager(session).get_group(temporary_group_name, create=True)
            if empty_group is None:
                raise RuntimeError("PseudobondManager.get_group() returned None")

            empty_in_tree = None
            for candidate in _suite_all_models(session):
                if candidate is empty_group:
                    empty_in_tree = candidate
                    break
            if empty_in_tree is None:
                # Some ChimeraX versions return a PseudobondGroup that is not
                # yet attached to the session model tree. Attach it explicitly.
                session.models.add([empty_group])
                for candidate in _suite_all_models(session):
                    if candidate is empty_group:
                        empty_in_tree = candidate
                        break

            if empty_in_tree is None:
                raise RuntimeError("empty pseudobond group could not be added to the session model tree")

            detected_id_str = _suite_model_spec(empty_in_tree)
            session.logger.info(
                f"[add_pi] No pi interactions found for {nested_model_spec}; created an empty interaction subgroup."
            )
        except Exception as e:
            session.logger.error(
                f"[add_pi] ERROR: No pseudobond group was created for {nested_model_spec}, "
                f"and the empty-group fallback failed: {e}"
            )
            return

    # Assign only the user-requested hierarchical ID. The pseudobond group
    # already has its final name from the moment it was created.
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
            f"[add_pi] ERROR: ChimeraX did not assign the generated interaction group to {nested_model_spec}. "
            "The requested model ID may already be occupied."
        )
        return

    # Keep the pseudobond group's original name. It is already the requested
    # chain-pair name, including for empty groups.
    renamed_group = assigned_model
    session.logger.info(
        f"[add_pi] Successfully assigned interaction model {nested_model_spec} "
        f"with name '{final_group_name}'."
    )

    # IMPORTANT: do not parent the centroid marker set under the pseudobond group.
    # The pseudobonds reference those markers, so making the markers children of
    # the pseudobond group creates a model-DAG cycle in ChimeraX. The centroid
    # marker set therefore remains a separate top-level model.

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
        output_filename = "pi.txt"
        try:
            with open(output_filename, "w") as f:
                f.write("\n".join(pbond_lines_output) + "\n\n")
                f.write("=" * 50 + "\n")
                f.write("      INTERFACE INTERACTION METRICS SUMMARY      \n")
                f.write("=" * 50 + "\n")
                f.write(f"Total number of pi-type interactions found: {pbond_count}\n")
                f.write("-" * 50 + "\n")
                for chain_name, count_val in sorted(chain_counts.items()):
                    f.write(f"  • Chain {chain_name} : {count_val} pi interaction contacts found.\n")
                f.write("=" * 50 + "\n")
            print(f"[add_pi] Successfully exported text mapping table to: {os.path.abspath(output_filename)}")
        except Exception as e:
            print(f"Error writing file: {e}")

    # Do not modify the source protein atom display/style state here. Other interaction
    # scripts (especially add_hbond) may use the same atoms. The pi script should
    # only create the pi pseudobonds/centroids and leave the molecular representation
    # entirely under the user's control.

    run(session, "select clear")
    
    if pbond_count == 0:
        session.logger.info("[add_pi] No potential pi-type interactions passed the snapshot cutoffs.")
        for subtype in ("pi-pi", "cation-pi", "sulfur-pi"):
            item = closest_candidates[subtype]
            if item is not None:
                d, r1, r2 = item
                session.logger.info(
                    f"[add_pi] Closest {subtype} candidate: {model_spec}/{r1.chain_id}:{r1.number} {r1.name} "
                    f"- {r2.chain_id}:{r2.number} {r2.name} = {d:.2f} A"
                )

    print("\n" + "="*50)
    print("      INTERFACE INTERACTION METRICS PER CHAIN      ")
    print("="*50)
    for chain_name, count_val in sorted(chain_counts.items()):
        print(f"  • Chain {chain_name} : {count_val} pi interaction contacts found.")
    print("="*50)
    print(f"Successfully generated exactly {pbond_count} single-line interactions inside group '{final_group_name}' at model '{nested_model_spec}'!\n")


def _parse_atom_or_residue_spec(spec):
    """Parse #MODEL/CHAIN:RES[@ATOM] into ids for lookup."""
    s = str(spec).strip()
    if not s.startswith('#'):
        raise ValueError(f"atom/residue spec must start with #, got {spec}")
    body = s[1:]
    if '/' not in body or ':' not in body:
        raise ValueError(f"expected #MODEL/CHAIN:RES[@ATOM], got {spec}")
    model_part, rest = body.split('/', 1)
    chain_part, res_atom = rest.split(':', 1)
    atom_name = None
    if '@' in res_atom:
        res_part, atom_name = res_atom.split('@', 1)
        atom_name = atom_name.strip()
    else:
        res_part = res_atom
    if not model_part or not all(x.isdigit() for x in model_part.split('.')):
        raise ValueError(f"invalid model in {spec}")
    if not chain_part or not res_part.strip() or not res_part.strip().lstrip('-').isdigit():
        raise ValueError(f"invalid chain/residue in {spec}")
    return tuple(int(x) for x in model_part.split('.')), chain_part.strip(), int(res_part.strip()), atom_name


def _find_model_by_id(session, id_tuple):
    wanted = tuple(id_tuple)
    for m in session.models:
        if tuple(getattr(m, 'id', ())) == wanted:
            return m
    return None


def _resolve_endpoint_atoms(struct_model, spec):
    """Return actual protein atoms matching a user spec; atom optional."""
    _, chain, resnum, atom_name = _parse_atom_or_residue_spec(spec)
    matches = []
    for a in struct_model.atoms:
        r = getattr(a, 'residue', None)
        if r is None:
            continue
        if str(r.chain_id).strip() != chain or int(r.number) != resnum:
            continue
        if atom_name is not None and str(a.name).strip() != atom_name:
            continue
        matches.append(a)
    return matches


def _find_pi_marker_for_residue(session, residue_spec, tolerance=0.15):
    """Find the centroid marker corresponding to an aromatic residue across pi marker sets."""
    id_tuple, chain, resnum, atom_name = _parse_atom_or_residue_spec(residue_spec)
    if atom_name is not None:
        raise ValueError("The centroid endpoint should be a residue spec without @ATOM")
    struct_model = _find_model_by_id(session, id_tuple)
    if struct_model is None or not hasattr(struct_model, 'atoms'):
        return []
    residue = None
    for a in struct_model.atoms:
        r = a.residue
        if str(r.chain_id).strip() == chain and int(r.number) == resnum:
            residue = r
            break
    if residue is None:
        return []
    ring_names = {
        'PHE': {'CG','CD1','CD2','CE1','CE2','CZ'},
        'TYR': {'CG','CD1','CD2','CE1','CE2','CZ'},
        'TRP': {'CD2','CE2','CE3','CZ2','CZ3','CH2'},
        'HIS': {'CG','ND1','CD2','CE1','NE2'},
    }
    names = ring_names.get(str(residue.name).upper())
    if not names:
        return []
    ring = [a for a in residue.atoms if a.name in names]
    if not ring:
        return []
    center = np.mean(np.array([a.coord for a in ring]), axis=0)
    hits=[]
    for m in session.models:
        if not hasattr(m, 'atoms') or not hasattr(m, 'name'):
            continue
        if not str(getattr(m,'name','')).startswith('pi_centroids_'):
            continue
        for marker in m.atoms:
            if np.linalg.norm(marker.coord-center) <= tolerance:
                hits.append(marker)
    return hits


def _toggle_pi_pseudobond(session, pi_group_spec, endpoint1, endpoint2, visible):
    """Hide/show the single pseudobond matching two protein/centroid endpoints."""
    group_id = str(pi_group_spec).strip()
    if group_id.startswith('#'):
        group_id = group_id[1:]
    if not group_id or any(x == '' or not x.isdigit() for x in group_id.split('.')):
        raise ValueError("PI group must be a model id such as #7")
    group = _find_model_by_id(session, tuple(int(x) for x in group_id.split('.')))
    if group is None or not hasattr(group, 'pseudobonds'):
        raise ValueError(f"Could not locate pseudobond group #{group_id}")

    parsed1 = _parse_atom_or_residue_spec(endpoint1)
    parsed2 = _parse_atom_or_residue_spec(endpoint2)
    protein_specs = []
    marker_spec = []
    for parsed, original in ((parsed1, endpoint1), (parsed2, endpoint2)):
        if parsed[3] is None:
            marker_spec.append(original)
        else:
            protein_specs.append(original)
    if len(protein_specs) == 2:
        protein_atoms = []
        for spec in protein_specs:
            model_id, _, _, _ = _parse_atom_or_residue_spec(spec)
            sm = _find_model_by_id(session, model_id)
            if sm is None:
                raise ValueError(f"Could not locate model for {spec}")
            atoms = _resolve_endpoint_atoms(sm, spec)
            if not atoms:
                raise ValueError(f"Could not locate atom {spec}")
            protein_atoms.append(atoms)
        a_candidates = [(a,b) for a in protein_atoms[0] for b in protein_atoms[1]]
    elif len(protein_specs) == 1 and len(marker_spec) == 1:
        model_id, _, _, _ = _parse_atom_or_residue_spec(protein_specs[0])
        sm = _find_model_by_id(session, model_id)
        if sm is None:
            raise ValueError(f"Could not locate model for {protein_specs[0]}")
        pats = _resolve_endpoint_atoms(sm, protein_specs[0])
        if not pats:
            raise ValueError(f"Could not locate atom {protein_specs[0]}")
        markers = _find_pi_marker_for_residue(session, marker_spec[0])
        a_candidates = [(a,m) for a in pats for m in markers]
    else:
        raise ValueError("Endpoint specs must identify two atoms or one atom and one aromatic residue centroid")

    matched = []
    for pb in group.pseudobonds:
        ends = tuple(pb.atoms)
        for a,b in a_candidates:
            if (ends[0] is a and ends[1] is b) or (ends[0] is b and ends[1] is a):
                matched.append(pb)
                break
    if not matched:
        raise ValueError(f"No pseudobond in {pi_group_spec} matches {endpoint1} <-> {endpoint2}")
    for pb in matched:
        pb.display = bool(visible)
    action = 'shown' if visible else 'hidden'
    session.logger.info(f"[add_pi] {action} {len(matched)} π pseudobond(s) in {pi_group_spec}: {endpoint1} <-> {endpoint2}")


def hide_pi_pseudobond(session, pi_group_spec, endpoint1, endpoint2):
    _toggle_pi_pseudobond(session, pi_group_spec, endpoint1, endpoint2, False)


def show_pi_pseudobond(session, pi_group_spec, endpoint1, endpoint2):
    _toggle_pi_pseudobond(session, pi_group_spec, endpoint1, endpoint2, True)

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


def add_pi(session, model_input, chain1_id, chain2_id, print_file_keyword, print_file_value, interaction_model_id, showLabels='yes'):
    chain2_list=_suite_chain_list(chain2_id)
    if not chain2_list:
        session.logger.error('[add_pi] No target chains were supplied.')
        return
    source=_suite_find_model(session,model_input)
    if source is None or not hasattr(source,'atoms'):
        session.logger.error(f'[add_pi] Could not locate coordinate model {model_input}.')
        return
    parent_spec=str(interaction_model_id).strip()
    if not parent_spec.startswith('#'):
        session.logger.error('[add_pi] Parent interaction model must look like #2.1.')
        return
    if _suite_find_model(session,parent_spec) is not None:
        session.logger.error(f'[add_pi] Parent model {parent_spec} already exists. Remove it before rebuilding this interaction family.')
        return

    source_id='_'.join(map(str,source.id))
    parent=Model(f'Model_{source_id}_pi',session)
    session.models.add([parent])
    run(session, f'rename {_suite_model_spec(parent)} name "Model_{source_id}_pi" id {parent_spec}')
    parent=_suite_find_model(session,parent_spec)
    if parent is None:
        session.logger.error(f'[add_pi] Could not create parent interaction model {parent_spec}.')
        return

    counts={}; created=[]
    for idx,c2 in enumerate(chain2_list,start=1):
        child_spec=parent_spec+'.'+str(idx)
        snap=_suite_snapshot(source)
        before_ids={id(m) for m in _suite_all_models(session)}
        try:
            _native_add_pi(session,_suite_model_spec(source),str(chain1_id),str(c2),'printFile','no',child_spec,showLabels)
        finally:
            _suite_restore(snap)

        child=_suite_find_model(session,child_spec)
        if child is None or not hasattr(child,'pseudobonds'):
            counts[c2]=0
            continue

        # _native_add_pi created the group with its final name. Renaming a
        # PseudobondGroup here can interfere with the manager's name registry.
        try:
            child._interaction_suite_kind='pi'
            child._interaction_suite_source_id=source_id
            child._interaction_suite_chain1=str(chain1_id)
            child._interaction_suite_chain2=str(c2)
        except Exception:
            pass
        _suite_hide_group(child)
        created.append(child)
        counts[c2]=len(list(getattr(child,'pseudobonds',())))

        if 'pi'=='pi':
            # Each chain-pair run gets its own centroid marker model.  Rename the
            # newly created marker set using the actual coordinate model and the
            # exact chain pair, rather than the interaction subgroup ID.  This
            # keeps centroid bookkeeping unambiguous for show_interactions.
            centroid_name = (
                f"pi_centroids_Model_{source_id}_Chain_{chain1_id}_to_Chain_{c2}"
            )
            for m in _suite_all_models(session):
                if str(getattr(m,'name','')).startswith('pi_centroids_') and id(m) not in before_ids:
                    try:
                        m.name = centroid_name
                    except Exception:
                        try:
                            run(session, f'rename {_suite_model_spec(m)} name "{centroid_name}"')
                        except Exception:
                            pass
                    try: m.display=False
                    except Exception: pass

    mode=str(print_file_value).strip().lower()
    total=sum(counts.values())
    print('\n'+'='*50)
    print('      INTERFACE INTERACTION METRICS PER CHAIN      ')
    print('='*50)
    for c2 in chain2_list:
        print(f'  - Chain {c2} : {counts.get(c2,0)} pi interaction contacts found.')
    print('='*50)
    print(f'Created parent {parent_spec} with {len(created)} chain-pair subgroup(s); total contacts: {total}')

    if mode in ('yes','true','on','1'):
        filename='pi.txt'
        with open(filename,'w') as f:
            f.write('='*50+'\n')
            f.write('      INTERFACE INTERACTION METRICS SUMMARY      \n')
            f.write('='*50+'\n')
            f.write('Total number of pi interactions found: '+str(total)+'\n')
            f.write('-'*50+'\n')
            for c2 in chain2_list:
                f.write(f'  - Chain {c2} : {counts.get(c2,0)} pi interaction contacts found.\n')
            f.write('='*50+'\n')


desc = CmdDesc(
    required=[("model_input", StringArg), ("chain1_id", StringArg), ("chain2_id", StringArg), ("print_file_keyword", StringArg), ("print_file_value", StringArg), ("interaction_model_id", StringArg)],
    keyword=[("showLabels", StringArg)],
    synopsis="Generate pi interaction subgroups by chain pair"
)
register("add_pi", desc, add_pi, logger=session.logger)
print("[add_pi] Command initialized successfully with chain-pair subgroups.")


# Preserve the existing add_pi helper commands.
hide_pi_desc = CmdDesc(
    required=[("pi_group_spec", StringArg), ("endpoint1", StringArg), ("endpoint2", StringArg)],
    synopsis="Hide a single pi-interaction pseudobond"
)
show_pi_desc = CmdDesc(
    required=[("pi_group_spec", StringArg), ("endpoint1", StringArg), ("endpoint2", StringArg)],
    synopsis="Show a single pi-interaction pseudobond"
)
register("hide_pi", hide_pi_desc, hide_pi_pseudobond, logger=session.logger)
register("show_pi", show_pi_desc, show_pi_pseudobond, logger=session.logger)
