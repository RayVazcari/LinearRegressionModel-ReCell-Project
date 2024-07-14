<h1><center><font size=10>Data Science and Business Analytics</center></font></h1>
<center>Project 3 - Supervised Learning: Dynamic Pricing for ReCell</center></h1><p
<center>Jorge Ramon Vazquez Campero</center></h1>

**`| Supervised Learning | Linear Regression | Data Visualization | Statistical Analysis | Python | Data Cleaning | Univariate Analysis | Bivariate Analysis | Multivariate Analysis | Insights Generation |`**

This project involved building a linear regression model to predict the price of used phones and tablets, based on various attributes. By analyzing the data and identifying significant factors, the goal was to develop a dynamic pricing strategy for ReCell, a startup in the refurbished device market. Throughout this project, I enhanced my skills in data preprocessing, statistical analysis, and machine learning model building, allowing me to uncover key insights and actionable recommendations for the business. This experience has significantly contributed to my growth as a data scientist, helping me to apply theoretical knowledge in a real-world scenario and gain valuable hands-on experience. 🚀

<p align="left"> 
  <a href="https://github.com/RayVazcari?tab=followers">
    <img alt="followers" title="Follow me on Github" src="https://custom-icon-badges.demolab.com/github/followers/RayVazcari?color=236ad3&labelColor=1155ba&style=for-the-badge&logo=person-add&label=Follow me on Github &logoColor=white"/></a>
  <a href="https://www.linkedin.com/in/rayvazcari/">
    <img alt="Linkedin Profile" title="Linkedin Profile" src="https://custom-icon-badges.demolab.com/badge/-Linkedin%20Profile-blue?style=for-the-badge&logoColor=white&logo=linkedin"/></a>
</p>

---

### 🧰 Languages and Tools I Used on This Project
<img align="left" alt="Jupyter" width="30px" style="padding-right:10px;" src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/jupyter/jupyter-original-wordmark.svg" />
<img align="left" alt="Maplotlib" width="30px" style="padding-right:10px;" src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/matplotlib/matplotlib-original.svg" />
<img align="left" alt="Numpy" width="30px" style="padding-right:10px;" src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/numpy/numpy-original.svg" />
<img align="left" alt="Pandas" width="30px" style="padding-right:10px;" src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/pandas/pandas-original.svg" />
<img align="left" alt="Plotly" width="30px" style="padding-right:10px;" src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/plotly/plotly-original.svg" />
<img align="left" alt="Python" width="30px" style="padding-right:10px;"  src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/python/python-original.svg" />
<img align="left" alt="Raspberry Pi" width="30px" style="padding-right:10px;"  src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/raspberrypi/raspberrypi-original.svg" />
<img align="left" alt="VScode" width="30px" style="padding-right:10px;"  src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/vscode/vscode-original.svg" />
<img align="left" alt="Seaborn" width="30px" style="padding-right:10px;" src="https://seaborn.pydata.org/_images/logo-mark-lightbg.svg"  /> 
<img align="left" alt="Maplotlib" width="30px" style="padding-right:10px;"  src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/pytorch/pytorch-original.svg" />

<br />


---

<center><img src="https://mobilerecell.com/wp-content/uploads/2024/01/mobile-recell-secondary-interim-footer.png"></center>

<br />

---

### 📊 Project Overview

## Problem Statement

### Business Context

Buying and selling used phones and tablets used to be something that happened on a handful of online marketplace sites. But the used and refurbished device market has grown considerably over the past decade, and a new IDC (International Data Corporation) forecast predicts that the used phone market would be worth $52.7bn by 2023 with a compound annual growth rate (CAGR) of 13.6% from 2018 to 2023. This growth can be attributed to an uptick in demand for used phones and tablets that offer considerable savings compared with new models.

Refurbished and used devices continue to provide cost-effective alternatives to both consumers and businesses that are looking to save money when purchasing one. There are plenty of other benefits associated with the used device market. Used and refurbished devices can be sold with warranties and can also be insured with proof of purchase. Third-party vendors/platforms, such as Verizon, Amazon, etc., provide attractive offers to customers for refurbished devices. Maximizing the longevity of devices through second-hand trade also reduces their environmental impact and helps in recycling and reducing waste. The impact of the COVID-19 outbreak may further boost this segment as consumers cut back on discretionary spending and buy phones and tablets only for immediate needs.

### Objective
The rising potential of this comparatively under-the-radar market fuels the need for an ML-based solution to develop a dynamic pricing strategy for used and refurbished devices. ReCell, a startup aiming to tap the potential in this market, has hired you as a data scientist. They want you to analyze the data provided and build a linear regression model to predict the price of a used phone/tablet and identify factors that significantly influence it.

### Data Description
The data contains the different attributes of used/refurbished phones and tablets. The data was collected in the year 2021. The detailed data dictionary is given below.

- `brand_name`: Name of manufacturing brand
- `os`: OS on which the device runs
- `screen_size`: Size of the screen in cm
- `4g`: Whether 4G is available or not
- `5g`: Whether 5G is available or not
- `main_camera_mp`: Resolution of the rear camera in megapixels
- `selfie_camera_mp`: Resolution of the front camera in megapixels
- `int_memory`: Amount of internal memory (ROM) in GB
- `ram`: Amount of RAM in GB
- `battery`: Energy capacity of the device battery in mAh
- `weight`: Weight of the device in grams
- `release_year`: Year when the device model was released
- `days_used`: Number of days the used/refurbished device has been used
- `normalized_new_price`: Normalized price of a new device of the same model in euros
- `normalized_used_price`: Normalized price of the used/refurbished device in euros

---

### Summary of Findings

1. **Brand and Resale Value:**
   - **Observation:** Devices from popular and recognizable brands like Apple and Samsung hold much higher resale value than other brands, even when the specs might not always be the same.
   - **Recommendation:** ReCell should develop a strategy for acquiring, refurbishing, and marketing devices from these high-value brands to maximize profitability.

2. **RAM and Device Performance:**
   - **Observation:** There is a very strong correlation between the amount of RAM in a device and its resale value.
   - **Recommendation:** Highlight RAM specifications in product listings and marketing materials. Offer high-RAM devices with upgraded configurations to attract performance-conscious customers.

3. **Camera Quality:**
   - **Observation:** Both main and selfie camera megapixels are positively correlated with the resale price.
   - **Recommendation:** Market the high-quality camera features of devices, especially targeting segments of customers who prioritize photography.

4. **Battery Capacity:**
   - **Observation:** Although battery capacity is not directly a determinant of the price in the final model, there is a strong correlation between weight and battery capacity.
   - **Recommendation:** Ensure all refurbished devices possess excellent battery conditions and focus on acquiring devices with larger batteries.

5. **Operating System (OS):**
   - **Observation:** iOS devices have better resale value compared to other operating systems.
   - **Recommendation:** Focus on refurbishing and selling more iOS devices. Provide value-added services like additional warranty or exclusivity of the Apple ecosystem.

### Significance of Predictors
- **Const**: Represents the intercept in the regression model.
- **Main Camera MP**: Higher megapixels in the main camera significantly impact the resale value of devices.
- **Selfie Camera MP**: Quality of the selfie camera is a strong selling point.
- **RAM**: More RAM allows for better multitasking and smoother operation, positively influencing the resale price.
- **Weight**: Indicates build quality and battery size.
- **Normalized New Price**: Represents the initial market value of the device.
- **Years Since Release**: Indicates how long the device has been on the market.
- **Brand Name (Various Brands)**: Highlights the influence of brand reputation on resale value.
- **Operating System (OS)**: Devices with iOS have higher resale value.
- **4G Availability**: Ensures faster internet connectivity, highly desired in modern devices.

### Key Takeaways for the Business
- **It's All About the Brand:** Premium brands command a premium price.
- **Highlighting Specifications:** Devices with premium features should be highlighted in marketing materials.
- **Customer Segmentation:** Knowing customer preferences enables focused marketing strategies.

### Conclusion and Business Recommendations
**Conclusion**
- The data analysis highlights key drivers of resale value, such as brand, RAM, and camera quality. Focusing on these factors can help attract high-value customers.

**Business Recommendations**
- **Inventory Strategy:** Stock high-value brand devices.
- **Marketing and Sales:** Highlight performance attributes in marketing materials.
- **Refurbishment Quality:** Maintain high standards for battery life and device condition.
- **Customer Targeting:** Design marketing campaigns for differentiated customer segments.
- **Value-Added Services:** Provide extended warranties and exclusive features.
- **Trade-In Incentives:** Offer attractive trade-in deals for premium feature devices.
- **Consumer Education:** Educate consumers on the benefits of higher RAM, better cameras, and longer battery life to justify higher prices.
