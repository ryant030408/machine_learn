import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# data import
pkmn = pd.read_csv('Pokemon.csv')

# look at data
# print(pkmn.head())

# drop gena nd legendary data
pkmn = pkmn.drop(['Generation', 'Legendary'], 1)

# create a scatterplot with two variables
# sns.jointplot(x='HP', y='Attack', data=pkmn)
# plt.show()

# boxplot with hp argument
# sns.boxplot(y='HP', data=pkmn)
# plt.show()

# boxplot with no arguments, plots all stats
# sns.boxplot(data=pkmn)
# plt.show()

# dont need # in graph so we drop it
pkmn = pkmn.drop(['Total', '#'],1)

# now its much cleaner
# sns.boxplot(data=pkmn)
# plt.show()

# now to make a swarm plot, using melt to come up with colors
pkmn = pd.melt(pkmn, id_vars=['Name', 'Type 1', 'Type 2'], var_name='Stat')

print(pkmn.head)
# now we went from 800 rows of data to 4800

# sns.swarmplot(x='Stat', y='value', data=pkmn, hue='Type 1')
# plt.show()

# clean up data
sns.set_style("whitegrid")
with sns.color_palette([
    "#8ED752", "#F95643", "#53AFFE", "#C3D221", "#BBBDAF",
    "#AD5CA2", "#F8E64E", "#F0CA42", "#F9AEFE", "#A35449",
    "#FB61B4", "#CDBD72", "#7673DA", "#66EBFF", "#8B76FF",
    "#8E6856", "#C3C1D7", "#75A4F9"], n_colors=18, desat=.9):
    plt.figure(figsize=(12,10))
    plt.ylim(0, 275)
    sns.swarmplot(x='Stat', y='value', data=pkmn, hue='Type 1', split=True, size=7)
    plt.legend(bbox_to_anchor=(1,1), loc=2, borderaxespad=0.)
plt.show()