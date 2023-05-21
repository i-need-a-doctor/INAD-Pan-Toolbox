import cv2, pandas
import seaborn as sns

sns.set_theme(style="ticks")

input = cv2.imread("mse_curve.csv")

dots = sns.load_dataset("dots")
dots = dots[dots["align"] == "dots"]

# Define the palette as a list to specify exact values
palette = sns.color_palette("rocket_r")
print(dots)


# Plot the lines on two facets
sns.relplot(
    data=dots,
    x="index",
    y="mse_value",
    col="align",
    hue="coherence",
    size="choice",
    kind="line",
    size_order=["T1", "T2"],
    palette=palette,
    height=5,
    aspect=1,
    facet_kws=dict(sharex=False),
).savefig("test.png")
 