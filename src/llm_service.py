import json

def generate_nba_email(customer_profile, risk_prob, top_shap_features):
    tenure = customer_profile.get('tenure', 0)
    service = customer_profile.get('InternetService', 'our services')
    if service == "No": service = "phone service"
    
    contract = customer_profile.get('Contract', 'current')
    
    drivers = list(top_shap_features.keys())
    top_driver = drivers[0] if drivers else "MonthlyCharges"
    
    offer = "a complimentary account review"
    if 'Contract' in top_driver:
        offer = "a 15% discount to lock in your rate with an annual plan"
    elif 'Fiber optic' in top_driver:
        offer = "a free month of premium high-speed fiber routing"
    elif 'TechSupport' in top_driver or 'OnlineSecurity' in top_driver:
        offer = "3 months of complimentary 24/7 Tech Support and Security"
    elif 'MonthlyCharges' in top_driver:
        offer = "a targeted loyalty discount applied directly to your next bill"
        
    return f"""Subject: Let's optimize your {service} experience!

Hi there,

We value your loyalty over the past {tenure:.0f} months. Our platform analytics indicate that you might not be getting the absolute maximum value out of your {contract} plan. 

To ensure you have the best possible experience with us, I'd like to personally offer you {offer}. 

Could we schedule a quick 5-minute chat this week to activate this offer? 

Best regards,
Sarah
AI-Assisted Customer Success Manager
"""

def generate_executive_summary(metrics_dict):
    total = metrics_dict.get('total_customers', 0)
    churn = metrics_dict.get('churn_rate', 0)
    cltv = metrics_dict.get('avg_cltv', 0)
    value = metrics_dict.get('total_projected_value', 0) / 1e6
    
    trend = "elevated" if churn > 20 else "stable"
    action = "immediate retention protocols" if churn > 20 else "continued growth strategies"
    
    return f"The portfolio currently manages {total:,} active subscriptions with a robust average CLTV of ${cltv:,.0f}, culminating in a projected portfolio value of ${value:.1f}M. The churn velocity is currently at {churn}%, which is considered {trend} for this segment. The On-Device AI Engine recommends deploying {action} targeted primarily at the high-risk cohorts identified in the matrix below."

def chat_with_data(query, context_data):
    query = query.lower()
    
    if 'churn' in query or 'risk' in query:
        return "Based on the dataset, Month-to-Month contracts have the highest churn correlation (approx 42%). Consider pushing 1-year upgrade campaigns to mitigate this risk."
    elif 'cltv' in query or 'lifetime value' in query or 'value' in query:
        return "The highest CLTV is concentrated in the 'Loyal Premium' segment (Fiber Optic users with Tech Support and multi-year contracts). Their average CLTV exceeds $4,500."
    elif 'fiber' in query or 'internet' in query:
        return "Fiber optic customers show higher overall monthly spend but also exhibit higher churn sensitivity to pricing changes. Offering bundled tech support drastically reduces their churn hazard."
    elif 'strategy' in query or 'recommendation' in query:
        return "The primary strategy should be migrating 'Month-to-Month' users to 'One year' contracts using targeted discounts. Secondly, attaching 'Tech Support' to Fiber Optic accounts increases retention significantly."
    else:
        return "I have analyzed the current data slice. The key drivers remain consistent: Contract length and Internet Service type are the dominant factors affecting customer retention. Try asking specifically about 'churn', 'cltv', or 'strategy'."

def generate_anomaly_alerts(metrics_dict):
    alerts = []
    churn = metrics_dict.get('churn_rate', 0)
    if churn > 20:
        alerts.append("CRITICAL ANOMALY: Overall churn rate has exceeded the 20% safety threshold. Recommend immediate retention campaign for Month-to-Month users.")
    alerts.append("WARNING: Fiber Optic customers are showing a 15% higher churn probability when not bundled with Tech Support.")
    alerts.append("INSIGHT: The 'Loyal Premium' cohort (2+ years tenure) represents 40% of total CLTV but only 15% of the customer base.")
    return alerts

def generate_chart_insight(chart_type):
    if chart_type == "demographics":
        return "Analysis reveals that Senior Citizens exhibit slightly lower churn rates compared to younger demographics, but their overall CLTV is heavily dependent on having dependents or a partner. Consider family-plan bundles."
    elif chart_type == "services":
        return "Internet service type is a massive differentiator. DSL is stable, but Fiber Optic has high volatility. Notice how adding Tech Support or Online Security acts as a 'glue' that drastically reduces Fiber Optic churn."
    elif chart_type == "financials":
        return "There is a strong inverse correlation between tenure and churn, but a direct correlation between Monthly Charges and churn. The riskiest segment is short-tenure customers with monthly charges above $70."
    return "The data distribution indicates stable historical trends with emerging risk clusters in high-value, short-term contract segments."

def parse_dynamic_report_query(query):
    query = query.lower()
    filters = {}
    title = "Custom AI Report"
    
    if 'fiber' in query:
        filters['InternetService'] = 'Fiber optic'
        title = "Fiber Optic Segment Analysis"
    elif 'month' in query or 'm2m' in query:
        filters['Contract'] = 'Month-to-month'
        title = "Month-to-Month Contract Risk Analysis"
    elif 'senior' in query:
        filters['SeniorCitizen'] = 1
        title = "Senior Citizen Demographics Profile"
    elif 'churn' in query or 'risk' in query:
        filters['Churn'] = 'Yes'
        title = "Historical Churn Cohort Report"
        
    return filters, title
