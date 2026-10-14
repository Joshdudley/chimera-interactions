# ChimeraX Protein Interface Interactions

Six ChimeraX detection commands plus `show_interactions` for inspecting contacts between a source protein chain and one or more partner chains. Each detection command creates a parent model with a subgroup for every requested chain pair, including pairs with zero contacts.

These scripts run **inside UCSF ChimeraX**. They are visualization and screening tools: a proximity match is a candidate interaction, not a measurement of binding energy.

## Scripts

| Script | ChimeraX command | Detection rule |
| --- | --- | --- |
| [`add_hydrophobic.py`](add_hydrophobic.py) | `add_hydrophobic` | Closest non-carbonyl carbon pair within 4.5 Å, with both residues drawn from ALA, VAL, LEU, ILE, MET, PRO, PHE, TRP, or TYR; one contact per residue pair. |
| [`add_hbond.py`](add_hbond.py) | `add_hbond` | Donor–acceptor heavy-atom distance at most 3.5 Å; closest contact per residue pair. Uses an attached hydrogen for the displayed endpoint when available. |
| [`add_salt_bridge.py`](add_salt_bridge.py) | `add_salt_bridge` | ASP/GLU side-chain oxygens near LYS/ARG side-chain nitrogens, within 4.0 Å; closest contact per residue pair. |
| [`add_polar.py`](add_polar.py) | `add_polar` | Closest eligible O/N/S contact within 4.0 Å. Includes standard amino-acid backbone N/O and selected side chains; excludes residue pairs qualifying as hydrogen bonds and conventional salt-bridge atom pairs. |
| [`add_pi.py`](add_pi.py) | `add_pi` | Aromatic ring–ring, cation–π, and sulfur–π candidates. Uses a 6.5 Å ring-centroid cutoff or a 5.5 Å nearest-ring-atom criterion for ring–ring contacts, and a 5.5 Å cutoff for aromatic partners. |
| [`add_bad_hydrophobic.py`](add_bad_hydrophobic.py) | `add_bad_hydrophobic` | Non-carbonyl carbon of a hydrophobic residue near an O/N atom across the interface, within 3.5 Å; closest contact per residue pair. A proximity flag, not proof that the contact is energetically unfavorable. |
| [`show_interactions.py`](show_interactions.py) | `show_interactions`, `hide_interactions` | Displays existing interaction lines and contacting sidechains; can filter chain pairs, retain extra residues, control distance labels and hide other chains. |
| [`export_interactions.py`](export_interactions.py) | `export_interactions` | Exports the existing interaction groups to CSV, including groups with zero contacts. |
| [`compare_interactions.py`](compare_interactions.py) | Terminal command | Compares the ChimeraX CSV with PDBe Arpeggio JSON by chain and residue pair. |

The categories are based on the scripts' specific atom lists and distance rules. The carbon filters above include backbone `CA` because they exclude only the carbonyl carbon `C`. Hydrogen bonds use a heavy-atom distance proxy rather than an angle criterion. The π detector does not require ring-plane orientation. Polar contacts are suppressed for a residue pair when it meets the hydrogen-bond rules, preventing a second polar line over that pair's hydrogen-bond line.

## Load the commands

1. Download this repository and open your protein structure in ChimeraX.
2. In the ChimeraX command line, run each script you want to use. For example, if the files are in `/path/to/chimera-interactions`:

   ```chimerax
   runscript /path/to/chimera-interactions/add_hydrophobic.py
   runscript /path/to/chimera-interactions/add_hbond.py
   runscript /path/to/chimera-interactions/add_salt_bridge.py
   runscript /path/to/chimera-interactions/add_polar.py
   runscript /path/to/chimera-interactions/add_pi.py
   runscript /path/to/chimera-interactions/add_bad_hydrophobic.py
   runscript /path/to/chimera-interactions/show_interactions.py
   ```

`runscript` registers each command in the current ChimeraX session. Loading a newer version of a script again can produce ChimeraX's informational “replacing existing command” message.

## Run an analysis

All six commands use the same argument order:

```text
COMMAND #SOURCE_MODEL SOURCE_CHAIN TARGET_CHAIN_OR_RANGE printFile yes|no #PARENT_MODEL [showLabels yes|no]
```

For a coordinate model `#2`, compare chain `A` with chains `B` through `E`:

```chimerax
add_hydrophobic #2 A B-E printFile no #2.1 showLabels no
add_hbond #2 A B-E printFile no #2.2 showLabels no
add_salt_bridge #2 A B-E printFile no #2.3 showLabels no
add_polar #2 A B-E printFile no #2.4 showLabels no
add_pi #2 A B-E printFile no #2.5 showLabels no
add_bad_hydrophobic #2 A B-E printFile no #2.6 showLabels no
```

For example, `add_pi ... #2.5` creates `#2.5.1` for A–B, `#2.5.2` for A–C, `#2.5.3` for A–D, and `#2.5.4` for A–E. The corresponding subgroup names are `Model_2_pi_Chain_A_to_Chain_B` through `Model_2_pi_Chain_A_to_Chain_E`. Rerunning any `add_*` command at its existing parent ID closes and rebuilds that interaction family, clears retained pseudobonds, and reuses the ID. For `add_pi`, old centroid markers for that source and chain are removed too. If the destination ID belongs to another model, the command stops without replacing it. Run `add_hbond` before `add_polar` when rebuilding both families so polar contacts can exclude residue pairs already drawn as hydrogen bonds.

`showLabels no` keeps distance labels hidden initially; omitting it defaults to `yes`. The scripts create distance-label models even when the labels are hidden, so they can be shown later. Generated chain-pair groups start hidden so you can reveal the contacts you want to inspect in ChimeraX. Zero-contact chain pairs still get an empty subgroup. Empty groups and existing groups now use ChimeraX's session pseudobond manager, so their parent structures can be closed and the analysis run again without the pseudobond-manager deletion error.

## Display interactions

After running the six detection commands, show all existing interactions for model `#2`:

```chimerax
show_interactions #2 display_Dist false
```

This also shows cartoons for chains with interactions, including chains hidden by an earlier filtered call. Use `show_interactions #2` if you want to preserve the current distance-label visibility. To show particular interfaces, use chain pairs such as `A-D` (A with D, not a range):

```chimerax
show_interactions #2 A-D A-E keep #2/D:8,44,70 #2/E:8,44,70 display_Dist false
show_interactions #2 A-B hide_other_chains false
hide_interactions #2
```

`keep` adds listed residues to the display even if they have no selected contacts. `display_Dist true|false` controls distance labels; omitting it preserves the current visibility of labels in selected groups. When restoring the full view at the end of an imaging script, use `show_interactions #2 display_Dist false` to explicitly leave distance labels off in the displayed interaction groups. For filtered calls, `hide_other_chains true` is the default and hides cartoons and atoms of other chains. Set it to `false` to leave other chains' current display alone. `hide_interactions #2` hides the existing interaction lines and protein atoms; it does not delete interactions.

To rerun a scripted scene with fixed model numbers, start with `close all` **before opening its structures**. This clears the previous run's models, so the newly opened structures receive the intended numbers. If ChimeraX has assigned a built-in `missing structure` submodel to an ID reserved for an interaction family (for example, `#4.1`), choose an unused interaction-family ID or close that specific submodel after opening the structure and before calling the corresponding `add_*` command. Closing it removes the missing-segment pseudobond display, not the protein atoms.

## Compare with PDBe Arpeggio

The `printFile yes` summaries from the six detection commands contain totals per chain. To compare individual residue pairs, export the existing pseudobonds from the ChimeraX session:

```chimerax
runscript /path/to/chimera-interactions/export_interactions.py
export_interactions #2 2_interactions.csv
save structure.cif models #2
```

`export_interactions` includes hidden contacts and an explicit row for each chain-pair group with zero contacts. It does not rerun the detectors. If the structure is already available as mmCIF, use the same underlying structure for Arpeggio. The current PDBe Arpeggio implementation accepts mmCIF input and produces JSON; run it separately, for example:

```bash
pdbe-arpeggio structure.cif -o arpeggio_output/
```

Then compare the exported CSV with Arpeggio's interaction JSON using ordinary Python (outside ChimeraX):

```bash
python3 compare_interactions.py 2_interactions.csv path/to/arpeggio.json --output interaction_comparison.csv
```

This writes `interaction_comparison.csv` and `interaction_comparison_summary.txt`. The statuses `both`, `chimera_only`, and `arpeggio_only` refer to a chain pair, residue pair, and interaction category; multiple Arpeggio atom contacts for the same pair count once. A difference is a candidate to inspect, since the programs use different chemical rules and geometry. `bad_hydrophobic` has no direct Arpeggio category and is excluded from the category comparison. The export warns when a saved π centroid cannot be mapped unambiguously to a protein residue; the reported contact total should be checked before interpreting a comparison. Chain IDs, residue numbers, and insertion codes must match between the ChimeraX export and the mmCIF used by Arpeggio.

**Hydrophobic comparison:** Arpeggio may label contacts involving residues outside the nine-residue list used by `add_hydrophobic`, including carbon contacts from LYS, GLU, ARG, HIS, THR, and GLN. Our detector requires *both* residues to be on its list. Consequently, Arpeggio-only hydrophobic residue pairs can reflect this deliberate selection rule rather than missed contacts within the script's defined scope. The comparator already counts each residue pair only once, regardless of the number of atom contacts Arpeggio reports.

## Output and interpretation

Use `printFile yes` to write an aggregate summary for that command in ChimeraX's current working directory: `hydrophobic.txt`, `hbond.txt`, `salt_bridge.txt`, `polar.txt`, `pi.txt`, or `bad_hydrophobic.txt`. Running the same command again with `printFile yes` overwrites its corresponding text file. The chain-pair wrapper writes a totals summary, rather than a full list of individual contacts.

Distances and residue eligibility are deliberately simple screening choices. Structures without explicit hydrogens can still yield candidate hydrogen bonds using donor–acceptor heavy atoms. Salt-bridge and π classifications depend on the atom definitions above; protonation states, missing atoms, unusual residues, and alternative conformations can change what is detected. Inspect individual candidates before drawing chemical conclusions.

## Quick troubleshooting

| Symptom | What to check |
| --- | --- |
| Source model not found | Use the exact coordinate model ID shown in ChimeraX, including `#`. |
| Destination parent belongs to another model | Choose an unused parent ID; the command will not replace an unrelated model. |
| `#4.1` belongs to `missing structure` | ChimeraX created that submodel when opening the structure. Free the ID or choose a different interaction-family ID before running `add_hydrophobic`. |
| Distance labels appear after restoring the full view | Specify `display_Dist false` on the final `show_interactions` call. |
| Closing a structure reports `group not in manager` | Replace all six `add_*` scripts with the versions that use `session.pb_manager`, then start a fresh ChimeraX window before rebuilding the scene. Existing groups created by older scripts may remain affected. |
| No contacts for a chain pair | Check its empty subgroup; the structure may have no contacts under the command's cutoff. |
| Command is unknown | Run the corresponding script with `runscript` in the current ChimeraX session. |

The ChimeraX model number in these examples is illustrative. Substitute your own source model, chain IDs, and unused destination IDs.
