# ChimeraX Protein Interface Interactions

Six ChimeraX commands for inspecting contacts between a source protein chain and one or more partner chains. Each command creates a parent model with a subgroup for every requested chain pair, including pairs with zero contacts.

These scripts run **inside UCSF ChimeraX**. They are visualization and screening tools: a proximity match is a candidate interaction, not a measurement of binding energy.

## Scripts

| Script | ChimeraX command | Detection rule |
| --- | --- | --- |
| [`add_hydrophobic.py`](add_hydrophobic.py) | `add_hydrophobic` | Closest non-carbonyl carbon pair between selected hydrophobic residues, within 4.5 Å; one contact per residue pair. |
| [`add_hbond.py`](add_hbond.py) | `add_hbond` | Donor–acceptor heavy-atom distance at most 3.5 Å; closest contact per residue pair. Uses an attached hydrogen for the displayed endpoint when available. |
| [`add_salt_bridge.py`](add_salt_bridge.py) | `add_salt_bridge` | ASP/GLU side-chain oxygens near LYS/ARG side-chain nitrogens, within 4.0 Å; closest contact per residue pair. |
| [`add_polar.py`](add_polar.py) | `add_polar` | Closest eligible O/N/S contact within 4.0 Å. Includes standard amino-acid backbone N/O and selected side chains; excludes residue pairs qualifying as hydrogen bonds and conventional salt-bridge atom pairs. |
| [`add_pi.py`](add_pi.py) | `add_pi` | Aromatic ring–ring, cation–π, and sulfur–π candidates. Uses a 6.5 Å ring-centroid cutoff or a 5.5 Å nearest-ring-atom criterion for ring–ring contacts, and a 5.5 Å cutoff for aromatic partners. |
| [`add_bad_hydrophobic.py`](add_bad_hydrophobic.py) | `add_bad_hydrophobic` | Non-carbonyl carbon of a hydrophobic residue near an O/N atom across the interface, within 3.5 Å; closest contact per residue pair. A proximity flag, not proof that the contact is energetically unfavorable. |

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

Each destination parent ID must be unused when its command runs. For example, `add_pi ... #2.5` creates `#2.5.1` for A–B, `#2.5.2` for A–C, `#2.5.3` for A–D, and `#2.5.4` for A–E. The corresponding subgroup names are `Model_2_pi_Chain_A_to_Chain_B` through `Model_2_pi_Chain_A_to_Chain_E`. To rebuild a family at the same IDs, close its existing parent model first or choose a new destination ID.

`showLabels no` keeps distance labels hidden initially; omitting it defaults to `yes`. Generated chain-pair groups start hidden so you can reveal the contacts you want to inspect in ChimeraX. Zero-contact chain pairs still get an empty subgroup.

## Output and interpretation

Use `printFile yes` to write an aggregate summary for that command in ChimeraX's current working directory: `hydrophobic.txt`, `hbond.txt`, `salt_bridge.txt`, `polar.txt`, `pi.txt`, or `bad_hydrophobic.txt`. Running the same command again with `printFile yes` overwrites its corresponding text file. The chain-pair wrapper writes a totals summary, rather than a full list of individual contacts.

Distances and residue eligibility are deliberately simple screening choices. Structures without explicit hydrogens can still yield candidate hydrogen bonds using donor–acceptor heavy atoms. Salt-bridge and π classifications depend on the atom definitions above; protonation states, missing atoms, unusual residues, and alternative conformations can change what is detected. Inspect individual candidates before drawing chemical conclusions.

## Quick troubleshooting

| Symptom | What to check |
| --- | --- |
| Source model not found | Use the exact coordinate model ID shown in ChimeraX, including `#`. |
| Destination parent already exists | Close that interaction parent or choose another unused parent ID. |
| No contacts for a chain pair | Check its empty subgroup; the structure may have no contacts under the command's cutoff. |
| Command is unknown | Run the corresponding script with `runscript` in the current ChimeraX session. |

The ChimeraX model number in these examples is illustrative. Substitute your own source model, chain IDs, and unused destination IDs.
