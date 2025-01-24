import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from utilities import j, Qb, aof, qo_vogel, qo_darcy, qo_standing, Qo, IPR_curve, IPR_curve_methods


# Data
Pr = 3000 #psia
Pb = 2300 # psia
Qt = 1500 #bpd
Pwft = 2400 #psia
THP = 360 #psia
wc = 0.9
API = 20
sg_h2o = 1.09
ID = 3.5 #in
tvd = 9000 #ft
md = 10500 #ft
C = 120

def pwf_darcy(q_test, pwf_test, q, pr, pb):
    pwf = pr - (q / j(q_test, pwf_test, pr, pb))
    return pwf

# Pwf when Pr < Pb (Saturated reservoir)
def pwf_vogel(q_test, pwf_test, q, pr, pb):
    pwf = 0.125 * pr * (-1 + np.sqrt(81 - 80 * q / aof(q_test, pwf_test, pr, pb)))
    return pwf

# Friction factor (f) from Hanzen-Williams equation
def f_darcy(Q, ID, C=120):
    f = (2.083 * (((100 * Q)/(34.3 * C))**1.85 * (1 / ID)**4.8655)) / 1000
    return f

# SGOil using API
def sg_oil(API):
    SG_oil = 141.5 / (131.5 + API)
    return SG_oil

# SG average of fluids
def sg_avg(API, wc, sg_h2o):
    sg_avg = wc * sg_h2o + (1-wc) * sg_oil(API)
    return sg_avg

# Average Gradient using fresh water gradient (0.433 psi/ft)
def gradient_avg(API, wc, sg_h2o):
    g_avg = sg_avg(API, wc, sg_h2o) * 0.433
    return g_avg


#df.columns

columns = ['Q(bpd)', 'Pwf(psia)', 'THP(psia)', 'Pgravity(psia)', 'f', 'F(ft)', 'Pf(psia)', 'Po(psia)', 'Psys(psia)']
df = pd.DataFrame(columns=columns)

# Here the AOF is divided per 10 in order to evaluate the pwf for these 10 different flow rates
df[columns[0]] = np.array([0, 750, 1400, 2250, 3000, 3750, 4500, 5250, 6000, 6750, 7500])
df[columns[1]] = df['Q(bpd)'].apply(lambda x: pwf_darcy(Qt, Pwft, x, Pr, Pb))
df[columns[2]] = THP
df[columns[3]] = gradient_avg(API, wc, sg_h2o) * tvd
df[columns[4]] = df['Q(bpd)'].apply(lambda x: f_darcy(x, ID, C))
df[columns[5]] = df['f'] * md
df[columns[6]] = gradient_avg(API, wc, sg_h2o) * df['F(ft)']
df[columns[7]] = df['THP(psia)'] + df['Pgravity(psia)'] + df['Pf(psia)']
df[columns[8]] = df['Po(psia)'] - df['Pwf(psia)']

#df

fig2, ax = plt.subplots(figsize=(18, 10))

ax.plot(df['Q(bpd)'].values, df['Pwf(psia)'].values, c='red', label='IPR')
ax.plot(df['Q(bpd)'].values, df['Po(psia)'].values, c='green', label='VLP')
ax.plot(df['Q(bpd)'].values, df['Psys(psia)'].values, c='b', label='System Curve')
ax.set_xlabel('Q(bpd)')
ax.set_ylabel('Pwf(psia)')
ax.set_xlim(0, df['Q(bpd)'].max() + 1000)
ax.set_title('Nodal Analysis')
ax.grid()
plt.legend()
plt.show()

fig = go.Figure()
fig.add_trace(go.Scatter(x=df['Q(bpd)'], y=df['Pwf(psia)'], name='IPR'))
fig.add_trace(go.Scatter(x=df['Q(bpd)'], y=df['Po(psia)'], name='VLP'))
fig.add_trace(go.Scatter(x=df['Q(bpd)'], y=df['Psys(psia)'], name='System Curve'))
fig.update_layout(title='Nodal Analysis')
fig.show()
