# GDL-DTI

## Pipeline

```mermaid
flowchart LR
    A[Raw \n Data] --> B[Processed \n Data]
    B --> C[Data \n Analysis]
    B --> D[Ground \n Truth]
    D --> E[Simulation \n Data]
    E --> F[Non-linear \n Least Squares]
    E --> G[Fully Connected \n Network]
    E --> H[Equivariant \n Network]
    G --> I[Fully Connected \n Network Analysis]
    H --> J[Equivariant \n Network Analysis]
    F --> K[Evaluation]
    I --> K
    J --> K
```

## Overview

<style type="text/css">
table{border-collapse:collapse;}
td, th{text-align:center;vertical-align:center;}
.bold{font-weight:bold;}
</style>

<table>
<tbody>
  <tr>
    <td class="bold" rowspan="3">Physics Equation</td>
    <td class="bold" colspan="5">Model</td>
  </tr>
  <tr>
    <td class="bold" rowspan="2">Non-linear<br>Least Squares</td>
    <td class="bold" colspan="3">Fully Connected</td>
    <td class="bold" rowspan="2">Equivariant</td>
  </tr>
  <tr>
    <td colspan="2">self-supervised</td>
    <td>supervised</td>
  </tr>
  <tr>
    <td>DTI</td>
    <td>Full Rank</td>
    <td>Full Rank</td>
    <td>-</td>
    <td>Full Rank</td>
    <td>Irreps</td>
  </tr>
  <tr>
    <td>IVIM-DTI</td>
    <td>Full Rank</td>
    <td>Full Rank<br></td>
    <td>Irreps<br></td>
    <td>- </td>
    <td>Irreps</td>
  </tr>
</tbody>
</table>

## Experiments

Tracking of experiments in `track.drawio` file.