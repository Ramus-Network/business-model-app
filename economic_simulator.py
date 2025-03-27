import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Collection Economics Simulator", page_icon="💰", layout="wide")

st.title("Collection Economics Simulator")
st.write("Explore the economic viability of a pay-per-query document retrieval service.")

# Sidebar for user inputs
st.sidebar.header("Cost Parameters")

# Scale selection for collection size
collection_scale = st.sidebar.radio(
    "Collection Size Scale",
    options=["Small (100-10K)", "Medium (1K-100K)", "Large (10K-1M)", "Enterprise (100K-10M)"],
    index=1
)

# Collection size with dynamic range based on scale selection
if collection_scale == "Small (100-10K)":
    collection_size = st.sidebar.slider(
        "Collection Size (documents)",
        min_value=100,
        max_value=10000,
        value=1000,
        step=100,
        format="%d"
    )
elif collection_scale == "Medium (1K-100K)":
    collection_size = st.sidebar.slider(
        "Collection Size (documents)",
        min_value=1000,
        max_value=100000,
        value=10000,
        step=1000,
        format="%d"
    )
elif collection_scale == "Large (10K-1M)":
    collection_size = st.sidebar.slider(
        "Collection Size (documents)",
        min_value=10000,
        max_value=1000000,
        value=100000,
        step=10000,
        format="%d"
    )
else:  # Enterprise
    collection_size = st.sidebar.slider(
        "Collection Size (documents)",
        min_value=100000,
        max_value=10000000,
        value=1000000,
        step=100000,
        format="%d"
    )

# Scale selection for query volume
user_scale = st.sidebar.radio(
    "User Base Scale",
    options=["Early Stage (10-100)", "Growing (100-1K)", "Established (1K-10K)", "Large (10K-100K)"],
    index=1
)

# User count with dynamic range based on scale selection
if user_scale == "Early Stage (10-100)":
    user_count = st.sidebar.slider(
        "Monthly Active Users",
        min_value=10,
        max_value=100,
        value=50,
        step=5,
        format="%d"
    )
elif user_scale == "Growing (100-1K)":
    user_count = st.sidebar.slider(
        "Monthly Active Users",
        min_value=100,
        max_value=1000,
        value=500,
        step=50,
        format="%d"
    )
elif user_scale == "Established (1K-10K)":
    user_count = st.sidebar.slider(
        "Monthly Active Users",
        min_value=1000,
        max_value=10000,
        value=5000,
        step=500,
        format="%d"
    )
else:  # Large
    user_count = st.sidebar.slider(
        "Monthly Active Users",
        min_value=10000,
        max_value=100000,
        value=50000,
        step=5000,
        format="%d"
    )

# User behavior
st.sidebar.subheader("User Behavior")

max_free_queries = st.sidebar.slider(
    "Maximum Free Queries Per User",
    min_value=1,
    max_value=50,
    value=10,
    step=1
)

avg_free_queries = st.sidebar.slider(
    "Average Free Queries Used Per User",
    min_value=0.1,
    max_value=float(max_free_queries),
    value=min(3.0, float(max_free_queries)),
    step=0.1,
    format="%.1f"
)

avg_paid_queries = st.sidebar.slider(
    "Average Paid Queries Per User",
    min_value=0.0,
    max_value=50.0,
    value=1.5,
    step=0.1,
    format="%.1f"
)

# Add monthly churn rate
monthly_churn_rate = st.sidebar.slider(
    "Monthly Churn Rate (%)",
    min_value=1.0,
    max_value=30.0,
    value=5.0,
    step=0.5,
    help="Percentage of users who leave each month"
) / 100  # Convert to decimal for calculations

# Calculate total queries based on user behavior
total_free_queries = user_count * avg_free_queries
total_paid_queries = user_count * avg_paid_queries
query_volume = total_free_queries + total_paid_queries

# Display calculated totals
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Total Monthly Queries:** {int(query_volume):,}")
st.sidebar.markdown(f"**Total Free Queries:** {int(total_free_queries):,}")
st.sidebar.markdown(f"**Total Paid Queries:** {int(total_paid_queries):,}")

# Business model parameters
st.sidebar.header("Business Model Parameters")

# Price range selection
price_range = st.sidebar.radio(
    "Price Range",
    options=["Micro ($0.001-$0.05)", "Standard ($0.01-$0.25)", "Premium ($0.05-$1.00)"],
    index=1
)

# Query price with dynamic range based on selection
if price_range == "Micro ($0.001-$0.05)":
    query_price = st.sidebar.number_input(
        "Query Price ($)",
        min_value=0.001,
        max_value=0.05,
        value=0.01,
        step=0.001,
        format="%.3f"
    )
elif price_range == "Standard ($0.01-$0.25)":
    query_price = st.sidebar.number_input(
        "Query Price ($)",
        min_value=0.01,
        max_value=0.25,
        value=0.10,
        step=0.01,
        format="%.2f"
    )
else:  # Premium
    query_price = st.sidebar.number_input(
        "Query Price ($)",
        min_value=0.05,
        max_value=1.00,
        value=0.25,
        step=0.05,
        format="%.2f"
    )

revenue_share = st.sidebar.slider(
    "Creator Revenue Share (%)",
    min_value=0,
    max_value=100,
    value=70,
    step=1
)

# Advanced parameters
st.sidebar.header("Advanced Parameters")
show_advanced = st.sidebar.checkbox("Show Advanced Parameters", value=False)

if show_advanced:
    clusters_per_document = st.sidebar.slider(
        "Clusters per Document",
        min_value=1,
        max_value=20,
        value=1,
        step=1,
        help="Number of semantic clusters per document. Higher values create finer-grained document representation."
    )
    
    chunks_per_cluster = st.sidebar.slider(
        "Chunks per Cluster",
        min_value=1,
        max_value=50,
        value=10,
        step=1
    )

    embedding_tokens_per_chunk = st.sidebar.slider(
        "Embedding Tokens per Chunk",
        min_value=128,
        max_value=1024,
        value=512,
        step=128
    )

    vector_dimensions = st.sidebar.slider(
        "Vector Dimensions",
        min_value=384,
        max_value=1536,
        value=768,
        step=128
    )

    input_tokens_per_query = st.sidebar.slider(
        "Input Tokens per Query",
        min_value=1000,
        max_value=50000,
        value=10000,
        step=1000
    )

    output_tokens_per_query = st.sidebar.slider(
        "Output Tokens per Query",
        min_value=10,
        max_value=1000,
        value=100,
        step=10
    )

    embedding_cost_per_m_tokens = st.sidebar.number_input(
        "Embedding Cost ($ per M tokens)",
        min_value=0.001,
        max_value=1.000,
        value=0.067,
        step=0.001,
        format="%.3f"
    )

    vector_storage_cost = st.sidebar.number_input(
        "Vector Storage Cost ($ per 100M dimensions)",
        min_value=0.01,
        max_value=1.00,
        value=0.05,
        step=0.01
    )

    query_cost_per_m_dimensions = st.sidebar.number_input(
        "Query Cost ($ per M dimensions)",
        min_value=0.001,
        max_value=0.100,
        value=0.010,
        step=0.001,
        format="%.3f"
    )

    ai_input_cost = st.sidebar.number_input(
        "AI Input Cost ($ per M tokens)",
        min_value=0.01,
        max_value=1.00,
        value=0.10,
        step=0.01
    )

    ai_output_cost = st.sidebar.number_input(
        "AI Output Cost ($ per M tokens)",
        min_value=0.01,
        max_value=5.00,
        value=0.40,
        step=0.01
    )

else:
    # Default values
    clusters_per_document = 1
    chunks_per_cluster = 10
    embedding_tokens_per_chunk = 512
    vector_dimensions = 768
    input_tokens_per_query = 10000
    output_tokens_per_query = 100
    embedding_cost_per_m_tokens = 0.067
    vector_storage_cost = 0.05
    query_cost_per_m_dimensions = 0.01
    ai_input_cost = 0.10
    ai_output_cost = 0.40
    
    # Default user lifetime parameters
    monthly_churn_rate = 0.05
    paid_user_percentage = 0.10

# Calculation function
def calculate_economics(collection_size, user_count, avg_free_queries, avg_paid_queries, query_price, revenue_share,
                       clusters_per_document, chunks_per_cluster, embedding_tokens_per_chunk, vector_dimensions,
                       input_tokens_per_query, output_tokens_per_query, embedding_cost_per_m_tokens,
                       vector_storage_cost, query_cost_per_m_dimensions, ai_input_cost, ai_output_cost, monthly_churn_rate):
    
    # Calculate total queries based on user behavior
    total_free_queries = user_count * avg_free_queries
    total_paid_queries = user_count * avg_paid_queries
    query_volume = total_free_queries + total_paid_queries
    
    # Calculate one-time embedding costs
    total_clusters = collection_size * clusters_per_document
    total_chunks = total_clusters * chunks_per_cluster
    total_embedding_tokens = total_chunks * embedding_tokens_per_chunk
    embedding_cost = total_embedding_tokens * embedding_cost_per_m_tokens / 1_000_000
    
    # Calculate monthly storage costs
    num_clusters = collection_size * clusters_per_document
    total_dimensions = num_clusters * vector_dimensions
    storage_cost_monthly = total_dimensions * vector_storage_cost / 100_000_000
    
    # Calculate query costs
    # Vector querying costs
    query_dimensions = (query_volume + num_clusters) * vector_dimensions
    vector_query_cost = query_dimensions * query_cost_per_m_dimensions / 1_000_000
    
    # AI costs
    ai_input_cost_monthly = (query_volume * input_tokens_per_query * ai_input_cost) / 1_000_000
    ai_output_cost_monthly = (query_volume * output_tokens_per_query * ai_output_cost) / 1_000_000
    total_ai_cost = ai_input_cost_monthly + ai_output_cost_monthly
    
    # Total query cost
    total_query_cost = vector_query_cost + total_ai_cost
    
    # Total monthly cost
    total_monthly_cost = storage_cost_monthly + total_query_cost
    
    # Amortized embedding cost (over 12 months)
    monthly_amortized_embedding = embedding_cost / 12
    total_cost_with_amortization = total_monthly_cost + monthly_amortized_embedding
    
    # Cost per query
    cost_per_query = total_monthly_cost / query_volume if query_volume > 0 else 0
    cost_per_query_with_amortization = total_cost_with_amortization / query_volume if query_volume > 0 else 0
    
    # Revenue calculations - only paid queries generate revenue
    total_revenue = total_paid_queries * query_price
    creator_revenue = total_revenue * (revenue_share / 100)
    platform_revenue = total_revenue - creator_revenue
    
    # Profit calculations
    profit = platform_revenue - total_monthly_cost
    profit_with_amortization = platform_revenue - total_cost_with_amortization
    profit_margin = (profit / platform_revenue * 100) if platform_revenue > 0 else 0
    profit_margin_with_amortization = (profit_with_amortization / platform_revenue * 100) if platform_revenue > 0 else 0
    
    # Break-even calculations
    platform_revenue_per_paid_query = query_price * (1 - revenue_share / 100)
    
    # Calculate how many paid queries needed to break even
    break_even_paid_queries = total_monthly_cost / platform_revenue_per_paid_query if platform_revenue_per_paid_query > 0 else float('inf')
    break_even_paid_queries_with_amortization = total_cost_with_amortization / platform_revenue_per_paid_query if platform_revenue_per_paid_query > 0 else float('inf')
    
    # Calculate break-even users based on current avg_paid_queries per user
    break_even_users = break_even_paid_queries / avg_paid_queries if avg_paid_queries > 0 else float('inf')
    break_even_users_with_amortization = break_even_paid_queries_with_amortization / avg_paid_queries if avg_paid_queries > 0 else float('inf')
    
    # LTV/CAC calculations
    # CAC: Customer Acquisition Cost
    # CAC = (total_free_queries * cost_per_query) / (user_count * monthly_churn_rate)
    cac = (total_free_queries * cost_per_query) / (user_count * monthly_churn_rate) if monthly_churn_rate > 0 else float('inf')
    
    # LTV: Lifetime Value
    # LTV = [(avg_paid_queries * query_price * (1 - revenue_share / 100)) - ((avg_free_queries + avg_paid_queries) * cost_per_query)] / monthly_churn_rate
    monthly_profit_per_user = (avg_paid_queries * query_price * (1 - revenue_share / 100)) - ((avg_free_queries + avg_paid_queries) * cost_per_query)
    ltv = monthly_profit_per_user / monthly_churn_rate if monthly_churn_rate > 0 else float('inf')
    
    # LTV/CAC Ratio
    ltv_cac_ratio = ltv / cac if cac > 0 else float('inf')
    
    return {
        "Collection Size": collection_size,
        "User Count": user_count,
        "Avg Free Queries Per User": avg_free_queries,
        "Avg Paid Queries Per User": avg_paid_queries,
        "Total Free Queries": total_free_queries,
        "Total Paid Queries": total_paid_queries,
        "Total Query Volume": query_volume,
        "Query Price": query_price,
        "Creator Revenue Share": revenue_share,
        
        "One-time Embedding Cost": embedding_cost,
        "Monthly Storage Cost": storage_cost_monthly,
        "Vector Query Cost": vector_query_cost,
        "AI Input Cost": ai_input_cost_monthly,
        "AI Output Cost": ai_output_cost_monthly,
        "Total Query Cost": total_query_cost,
        "Total Monthly Cost": total_monthly_cost,
        "Monthly Amortized Embedding": monthly_amortized_embedding,
        "Total Monthly Cost with Amortization": total_cost_with_amortization,
        
        "Cost Per Query": cost_per_query,
        "Cost Per Query with Amortization": cost_per_query_with_amortization,
        
        "Total Revenue": total_revenue,
        "Creator Revenue": creator_revenue,
        "Platform Revenue": platform_revenue,
        
        "Profit": profit,
        "Profit with Amortization": profit_with_amortization,
        "Profit Margin (%)": profit_margin,
        "Profit Margin with Amortization (%)": profit_margin_with_amortization,
        
        "Break-even Paid Queries": break_even_paid_queries,
        "Break-even Paid Queries with Amortization": break_even_paid_queries_with_amortization,
        "Break-even Users": break_even_users,
        "Break-even Users with Amortization": break_even_users_with_amortization,
        
        "CAC": cac,
        "LTV": ltv,
        "LTV/CAC Ratio": ltv_cac_ratio
    }

# Run the calculation
results = calculate_economics(
    collection_size, user_count, avg_free_queries, avg_paid_queries, query_price, revenue_share,
    clusters_per_document, chunks_per_cluster, embedding_tokens_per_chunk, vector_dimensions,
    input_tokens_per_query, output_tokens_per_query, embedding_cost_per_m_tokens,
    vector_storage_cost, query_cost_per_m_dimensions, ai_input_cost, ai_output_cost, monthly_churn_rate
)

# Create tabs for different sections
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Summary", "Cost Breakdown", "Revenue Analysis", "Breakeven Analysis", "Customer Economics"])

with tab1:
    st.header("Economic Summary")
    
    # Create three columns for high-level metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Monthly Profit", f"${results['Profit']:.2f}")
        st.metric("Profit Margin", f"{results['Profit Margin (%)']:.1f}%")
        st.metric("Cost Per Query", f"${results['Cost Per Query']:.4f}")
    
    with col2:
        st.metric("Monthly Revenue", f"${results['Total Revenue']:.2f}")
        st.metric("Creator Revenue", f"${results['Creator Revenue']:.2f}")
        st.metric("Platform Revenue", f"${results['Platform Revenue']:.2f}")
    
    with col3:
        st.metric("Total Monthly Cost", f"${results['Total Monthly Cost']:.2f}")
        st.metric("One-time Embedding Cost", f"${results['One-time Embedding Cost']:.2f}")
        st.metric("Break-even Users", f"{results['Break-even Users']:.0f}", 
                 help="Number of monthly active users needed to break even with current user behavior")
    
    # Add LTV/CAC Metrics in a new section with shorter labels
    st.subheader("Customer Economics")
    
    ltv_cac_cols = st.columns(3)
    
    with ltv_cac_cols[0]:
        st.metric("CAC", f"${results['CAC']:.2f}", 
                 help="Cost of free queries given away to acquire a new customer")
    
    with ltv_cac_cols[1]:
        st.metric("LTV", f"${results['LTV']:.2f}", 
                 help="Total profit a customer generates over their lifetime")
    
    with ltv_cac_cols[2]:
        ltv_cac_delta = None
        ltv_cac_color = "normal"
        
        if results['LTV/CAC Ratio'] < 1:
            ltv_cac_delta = "Unprofitable"
            ltv_cac_color = "inverse"
        elif results['LTV/CAC Ratio'] < 3:
            ltv_cac_delta = "Break-even"
            ltv_cac_color = "off"
        else:
            ltv_cac_delta = "Sustainable"
            ltv_cac_color = "normal"
        
        st.metric("LTV/CAC Ratio", f"{results['LTV/CAC Ratio']:.2f}", 
                 delta=ltv_cac_delta, delta_color=ltv_cac_color,
                 help="Ratio of LTV to CAC. Target > 3 for a sustainable business model")
    
    # Viability indicator
    st.subheader("Viability Assessment")
    
    if results['Profit'] <= 0:
        st.error("⚠️ This model is not profitable with current parameters!")
    elif results['Profit with Amortization'] <= 0:
        st.warning("⚠️ This model is profitable but does not cover embedding costs within 12 months.")
    else:
        st.success("✅ This model is profitable and sustainable!")
    
    # Key insights
    st.subheader("Key Insights")
    
    insights = []
    
    # Insight about fixed vs variable costs
    fixed_cost_percentage = (results['Monthly Storage Cost'] / results['Total Monthly Cost']) * 100
    insights.append(f"• Fixed costs represent {fixed_cost_percentage:.1f}% of monthly costs, making this model scalable with usage.")
    
    # Insight about user base and break-even
    if user_count < results['Break-even Users']:
        insights.append(f"• Current user base ({user_count:,}) is below break-even ({results['Break-even Users']:.0f}). Increase active users to achieve profitability.")
    else:
        users_to_breakeven_percent = (results['Break-even Users'] / user_count) * 100
        insights.append(f"• Only {users_to_breakeven_percent:.1f}% of current user base is needed to break even.")
    
    # Insight about LTV/CAC ratio
    if results['LTV/CAC Ratio'] < 1:
        insights.append(f"• LTV/CAC ratio ({results['LTV/CAC Ratio']:.2f}) is less than 1, indicating an unsustainable business model. Consider reducing free queries, lowering costs, or increasing prices.")
    elif results['LTV/CAC Ratio'] < 3:
        insights.append(f"• LTV/CAC ratio ({results['LTV/CAC Ratio']:.2f}) is positive but below the recommended value of 3. Consider optimizing free query allowance or increasing paid query conversion.")
    else:
        insights.append(f"• LTV/CAC ratio ({results['LTV/CAC Ratio']:.2f}) indicates a sustainable customer acquisition strategy with good lifetime value.")
    
    # Insight about churn rate
    if monthly_churn_rate > 0.10:
        insights.append(f"• Monthly churn rate ({monthly_churn_rate*100:.1f}%) is high. Reducing churn would significantly improve LTV and overall economics.")
    
    # Insight about revenue share
    if revenue_share > 80:
        insights.append(f"• Creator revenue share ({revenue_share}%) is very high. Consider a lower share for better platform economics.")
    elif revenue_share < 50:
        insights.append(f"• Creator revenue share ({revenue_share}%) is quite low. Consider increasing to attract quality content creators.")
    
    # Insight about query price
    if results['Cost Per Query'] > query_price * 0.5:
        insights.append(f"• Cost per query (${results['Cost Per Query']:.4f}) is more than 50% of price (${query_price:.2f}). Consider raising prices.")
    
    for insight in insights:
        st.write(insight)

with tab2:
    st.header("Cost Breakdown")
    
    # Create a DataFrame for cost components
    cost_components = {
        "Cost Component": ["Storage Cost", "Vector Query Cost", "AI Input Cost", "AI Output Cost", "Amortized Embedding Cost"],
        "Monthly Cost": [
            results["Monthly Storage Cost"],
            results["Vector Query Cost"],
            results["AI Input Cost"],
            results["AI Output Cost"],
            results["Monthly Amortized Embedding"]
        ]
    }
    cost_df = pd.DataFrame(cost_components)
    
    # Create a bar chart for costs
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0']
    ax.bar(cost_df["Cost Component"], cost_df["Monthly Cost"], color=colors)
    ax.set_ylabel("Monthly Cost ($)")
    ax.set_title("Monthly Cost Components")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Display the chart
    st.pyplot(fig)
    
    # Create a pie chart
    fig2, ax2 = plt.subplots(figsize=(8, 8))
    ax2.pie(cost_df["Monthly Cost"], labels=cost_df["Cost Component"], autopct='%1.1f%%', 
            startangle=90, colors=colors)
    ax2.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    ax2.set_title("Cost Distribution")
    
    # Display the pie chart
    st.pyplot(fig2)
    
    # Detailed cost table
    st.subheader("Detailed Cost Breakdown")
    
    detailed_costs = {
        "Cost Component": [
            "Total Documents",
            "Total Clusters",
            "Total Chunks",
            "One-time Embedding Cost",
            "Monthly Storage Cost",
            "Vector Query Cost",
            "AI Input Cost",
            "AI Output Cost",
            "Total Query Cost",
            "Total Monthly Cost",
            "Monthly Amortized Embedding",
            "Total Monthly Cost with Amortization"
        ],
        "Amount": [
            f"{collection_size:,}",
            f"{collection_size * clusters_per_document:,}",
            f"{collection_size * clusters_per_document * chunks_per_cluster:,}",
            f"${results['One-time Embedding Cost']:.2f}",
            f"${results['Monthly Storage Cost']:.2f}",
            f"${results['Vector Query Cost']:.2f}",
            f"${results['AI Input Cost']:.2f}",
            f"${results['AI Output Cost']:.2f}",
            f"${results['Total Query Cost']:.2f}",
            f"${results['Total Monthly Cost']:.2f}",
            f"${results['Monthly Amortized Embedding']:.2f}",
            f"${results['Total Monthly Cost with Amortization']:.2f}"
        ]
    }
    
    st.table(pd.DataFrame(detailed_costs))
    
    st.subheader("Per-Query Costs")
    st.write(f"Cost per query: ${results['Cost Per Query']:.6f}")
    st.write(f"Cost per query with embedding amortization: ${results['Cost Per Query with Amortization']:.6f}")

with tab3:
    st.header("Revenue Analysis")
    
    # Revenue flow
    st.subheader("Revenue Flow")
    
    # Create a DataFrame for revenue components
    revenue_components = {
        "Component": ["Total Revenue", "Creator Revenue", "Platform Revenue", "Total Cost", "Profit"],
        "Amount": [
            results["Total Revenue"],
            results["Creator Revenue"],
            results["Platform Revenue"],
            results["Total Monthly Cost"],
            results["Profit"]
        ]
    }
    revenue_df = pd.DataFrame(revenue_components)
    
    # Create a waterfall chart for revenue flow
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    
    # Plot the total revenue bar
    ax3.bar(0, revenue_df.loc[0, "Amount"], color='#66b3ff', width=0.5)
    ax3.text(0, revenue_df.loc[0, "Amount"]/2, f"${revenue_df.loc[0, 'Amount']:.2f}", 
            ha='center', va='center', color='white', fontweight='bold')
    
    # Plot creator revenue as a negative bar
    ax3.bar(1, -revenue_df.loc[1, "Amount"], bottom=revenue_df.loc[0, "Amount"], color='#ff9999', width=0.5)
    ax3.text(1, revenue_df.loc[0, "Amount"] - revenue_df.loc[1, "Amount"]/2, 
            f"-${revenue_df.loc[1, 'Amount']:.2f}", ha='center', va='center', color='white', fontweight='bold')
    
    # Plot platform revenue
    ax3.bar(2, revenue_df.loc[2, "Amount"], color='#99ff99', width=0.5)
    ax3.text(2, revenue_df.loc[2, "Amount"]/2, f"${revenue_df.loc[2, 'Amount']:.2f}", 
            ha='center', va='center', color='white', fontweight='bold')
    
    # Plot costs as a negative bar
    ax3.bar(3, -revenue_df.loc[3, "Amount"], bottom=revenue_df.loc[2, "Amount"], color='#ffcc99', width=0.5)
    ax3.text(3, revenue_df.loc[2, "Amount"] - revenue_df.loc[3, "Amount"]/2, 
            f"-${revenue_df.loc[3, 'Amount']:.2f}", ha='center', va='center', color='white', fontweight='bold')
    
    # Plot profit
    ax3.bar(4, revenue_df.loc[4, "Amount"], color='#c2c2f0', width=0.5)
    ax3.text(4, revenue_df.loc[4, "Amount"]/2, f"${revenue_df.loc[4, 'Amount']:.2f}", 
            ha='center', va='center', color='white', fontweight='bold')
    
    ax3.set_xticks(range(5))
    ax3.set_xticklabels(revenue_df["Component"])
    ax3.set_title("Revenue Flow")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Display the chart
    st.pyplot(fig3)
    
    # Revenue metrics
    st.subheader("Revenue Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Revenue Per Paid Query", f"${query_price:.2f}")
        st.metric("Creator Revenue Per Query", f"${query_price * (revenue_share/100):.2f}")
        st.metric("Platform Revenue Per Query", f"${query_price * (1-revenue_share/100):.2f}")
    
    with col2:
        paying_users_percentage = (results["Total Paid Queries"] / results["Total Query Volume"]) * 100 if results["Total Query Volume"] > 0 else 0
        st.metric("Paying Queries %", f"{paying_users_percentage:.1f}%")
        st.metric("Free Queries", f"{results['Total Free Queries']}")
        st.metric("Paid Queries", f"{results['Total Paid Queries']}")
    
    with col3:
        st.metric("Profit Per Paid Query", f"${results['Profit'] / results['Total Paid Queries']:.4f}" if results['Total Paid Queries'] > 0 else "$0.00")
        st.metric("Profit Margin", f"{results['Profit Margin (%)']:.1f}%")
        st.metric("Profit with Amortization", f"${results['Profit with Amortization']:.2f}")

with tab4:
    st.header("Breakeven Analysis")
    
    # Breakeven metrics
    st.subheader("Breakeven Points")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Break-even Users", f"{results['Break-even Users']:.0f}", 
                 help="Number of monthly active users needed to break even with current user behavior")
        st.metric("Break-even Paid Queries", f"{results['Break-even Paid Queries']:.0f}",
                 help="Number of paid queries needed to break even")
    
    with col2:
        st.metric("Break-even Users with Amortization", f"{results['Break-even Users with Amortization']:.0f}",
                 help="Number of monthly active users needed to break even including embedding cost amortization")
        st.metric("Break-even Paid Queries with Amortization", f"{results['Break-even Paid Queries with Amortization']:.0f}",
                 help="Number of paid queries needed to break even including embedding cost amortization")
    
    # Breakeven analysis
    st.subheader("User Volume Sensitivity")
    
    # Calculate profit for different user counts
    user_range = np.logspace(1, 5, 100)  # from 10 to 100,000 users
    profits = []
    profits_with_amortization = []

    for u in user_range:
        # Scale query volume while maintaining the same ratio of free/paid queries per user
        # Use the existing function to calculate economics for each user count
        temp_results = calculate_economics(
            collection_size, u, avg_free_queries, avg_paid_queries, query_price, revenue_share,
            clusters_per_document, chunks_per_cluster, embedding_tokens_per_chunk, vector_dimensions,
            input_tokens_per_query, output_tokens_per_query, embedding_cost_per_m_tokens,
            vector_storage_cost, query_cost_per_m_dimensions, ai_input_cost, ai_output_cost, monthly_churn_rate
        )
        
        profits.append(temp_results["Profit"])
        profits_with_amortization.append(temp_results["Profit with Amortization"])

    # Plot the profit curve
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    ax4.plot(user_range, profits, label="Profit", color="#66b3ff", linewidth=2)
    ax4.plot(user_range, profits_with_amortization, label="Profit with Amortization", color="#ff9999", linewidth=2, linestyle="--")

    # Add breakeven line
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax4.axvline(x=results["Break-even Users"], color='#66b3ff', linestyle='--', alpha=0.5)
    ax4.axvline(x=results["Break-even Users with Amortization"], color='#ff9999', linestyle='--', alpha=0.5)

    # Add current user count marker
    ax4.scatter([user_count], [results["Profit"]], color="#66b3ff", s=100, zorder=5)
    ax4.scatter([user_count], [results["Profit with Amortization"]], color="#ff9999", s=100, zorder=5)

    # Add annotations for break-even points
    ax4.annotate(f'Break-even: {results["Break-even Users"]:.0f} users', 
                 xy=(results["Break-even Users"], 0),
                 xytext=(results["Break-even Users"]*1.2, max(profits)/10),
                 arrowprops=dict(facecolor='#66b3ff', shrink=0.05, alpha=0.7),
                 color='#66b3ff')
    
    ax4.annotate(f'Break-even with amort.: {results["Break-even Users with Amortization"]:.0f} users', 
                 xy=(results["Break-even Users with Amortization"], 0),
                 xytext=(results["Break-even Users with Amortization"]*1.2, -max(profits)/10),
                 arrowprops=dict(facecolor='#ff9999', shrink=0.05, alpha=0.7),
                 color='#ff9999')

    ax4.set_xscale('log')
    ax4.set_xlabel('Number of Monthly Active Users')
    ax4.set_ylabel('Monthly Profit ($)')
    ax4.set_title('Profit vs User Count with Break-even Points')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Display the chart
    st.pyplot(fig4)

    # Price sensitivity
    st.subheader("Price Sensitivity")

    # Calculate profit for different price points
    price_range = np.linspace(0.01, 0.5, 50)  # from $0.01 to $0.50 per query
    price_profits = []
    price_margins = []

    for p in price_range:
        # Use the existing function to calculate economics for each price point
        temp_results = calculate_economics(
            collection_size, user_count, avg_free_queries, avg_paid_queries, p, revenue_share,
            clusters_per_document, chunks_per_cluster, embedding_tokens_per_chunk, vector_dimensions,
            input_tokens_per_query, output_tokens_per_query, embedding_cost_per_m_tokens,
            vector_storage_cost, query_cost_per_m_dimensions, ai_input_cost, ai_output_cost, monthly_churn_rate
        )
        
        price_profits.append(temp_results["Profit with Amortization"])
        price_margins.append(temp_results["Profit Margin with Amortization (%)"])

    # Create a figure with two subplots
    fig5, (ax5a, ax5b) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot profit vs price
    ax5a.plot(price_range, price_profits, color="#66b3ff", linewidth=2)
    ax5a.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax5a.scatter([query_price], [results["Profit with Amortization"]], color="red", s=100, zorder=5)
    ax5a.set_xlabel('Query Price ($)')
    ax5a.set_ylabel('Monthly Profit ($)')
    ax5a.set_title('Profit vs Price')
    ax5a.grid(True, alpha=0.3)

    # Plot margin vs price
    ax5b.plot(price_range, price_margins, color="#ff9999", linewidth=2)
    ax5b.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax5b.scatter([query_price], [results["Profit Margin with Amortization (%)"]], color="red", s=100, zorder=5)
    ax5b.set_xlabel('Query Price ($)')
    ax5b.set_ylabel('Profit Margin (%)')
    ax5b.set_title('Profit Margin vs Price')
    ax5b.grid(True, alpha=0.3)

    plt.tight_layout()

    # Display the chart
    st.pyplot(fig5)

with tab5:
    st.header("Customer Economics")
    
    # Create three columns for high-level metrics
    ltv_cac_cols = st.columns(3)
    
    with ltv_cac_cols[0]:
        st.metric("CAC", f"${results['CAC']:.2f}", 
                 help="Cost of free queries given away to acquire a new customer")
    
    with ltv_cac_cols[1]:
        st.metric("LTV", f"${results['LTV']:.2f}", 
                 help="Total profit a customer generates over their lifetime")
    
    with ltv_cac_cols[2]:
        ltv_cac_delta = None
        ltv_cac_color = "normal"
        
        if results['LTV/CAC Ratio'] < 1:
            ltv_cac_delta = "Unprofitable"
            ltv_cac_color = "inverse"
        elif results['LTV/CAC Ratio'] < 3:
            ltv_cac_delta = "Break-even"
            ltv_cac_color = "off"
        else:
            ltv_cac_delta = "Sustainable"
            ltv_cac_color = "normal"
        
        st.metric("LTV/CAC Ratio", f"{results['LTV/CAC Ratio']:.2f}", 
                 delta=ltv_cac_delta, delta_color=ltv_cac_color,
                 help="Ratio of LTV to CAC. Target > 3 for a sustainable business model")
    
    # Add explanation of CAC and LTV calculations
    st.subheader("How CAC and LTV are calculated")
    
    st.markdown("""
    ### Customer Acquisition Cost (CAC)
    
    **Formula:** CAC = (total_free_queries * cost_per_query) / (user_count * monthly_churn_rate)
    
    This formula calculates the cost to acquire a new customer by:
    - Taking the total cost of all free queries given away
    - Dividing by the number of new customers gained (in a steady state, new customers = churned customers)
    
    ### Lifetime Value (LTV)
    
    **Formula:** LTV = monthly_profit_per_user / monthly_churn_rate
    
    Where:
    - monthly_profit_per_user = (avg_paid_queries * query_price * (1 - revenue_share / 100)) - ((avg_free_queries + avg_paid_queries) * cost_per_query)
    
    This formula calculates the total profit a customer generates over their lifetime by:
    - Calculating the monthly profit from each user (revenue from paid queries minus costs of all queries)
    - Dividing by the monthly churn rate (since average customer lifetime = 1 / churn_rate)
    
    ### LTV/CAC Ratio
    
    **LTV/CAC Ratio = LTV / CAC**
    
    This ratio indicates the sustainability of your customer acquisition strategy:
    - Ratio < 1: Unsustainable (losing money on each customer)
    - Ratio 1-3: Breaking even but may not be sustainable long-term
    - Ratio > 3: Sustainable business model with good customer economics
    """)
    
    # Add a numerical example using current values
    st.subheader("Numerical example with current values")
    
    # Format current values nicely for display
    example_text = f"""
    **Current values:**
    - Monthly active users: {user_count:,}
    - Monthly churn rate: {monthly_churn_rate*100:.1f}%
    - Average free queries per user: {avg_free_queries:.1f}
    - Average paid queries per user: {avg_paid_queries:.1f}
    - Query price: ${query_price:.2f}
    - Cost per query: ${results['Cost Per Query']:.4f}
    - Platform revenue share: {100-revenue_share:.0f}%
    
    **CAC calculation:**
    CAC = ({total_free_queries:,.0f} * ${results['Cost Per Query']:.4f}) / ({user_count:,} * {monthly_churn_rate:.2f})
    CAC = ${total_free_queries * results['Cost Per Query']:.2f} / {user_count * monthly_churn_rate:.0f}
    CAC = ${results['CAC']:.2f}
    
    **LTV calculation:**
    Monthly profit per user = ({avg_paid_queries:.1f} * ${query_price:.2f} * {(1-revenue_share/100):.2f}) - (({avg_free_queries:.1f} + {avg_paid_queries:.1f}) * ${results['Cost Per Query']:.4f})
    Monthly profit per user = ${(avg_paid_queries * query_price * (1-revenue_share/100)):.2f} - ${((avg_free_queries + avg_paid_queries) * results['Cost Per Query']):.2f}
    Monthly profit per user = ${(avg_paid_queries * query_price * (1-revenue_share/100)) - ((avg_free_queries + avg_paid_queries) * results['Cost Per Query']):.2f}
    
    LTV = ${(avg_paid_queries * query_price * (1-revenue_share/100)) - ((avg_free_queries + avg_paid_queries) * results['Cost Per Query']):.2f} / {monthly_churn_rate:.2f}
    LTV = ${results['LTV']:.2f}
    
    **LTV/CAC Ratio:**
    LTV/CAC = ${results['LTV']:.2f} / ${results['CAC']:.2f}
    LTV/CAC = {results['LTV/CAC Ratio']:.2f}
    """
    
    st.markdown(example_text)
    
    # Add visualization
    st.subheader("LTV/CAC Sensitivity Analysis")
    
    # Create column selector for the sensitivity analysis
    sensitivity_param = st.selectbox(
        "Select parameter for sensitivity analysis:",
        ["Monthly Churn Rate", "Average Free Queries", "Average Paid Queries", "Query Price"]
    )
    
    # Set up ranges for different parameters
    if sensitivity_param == "Monthly Churn Rate":
        param_range = np.linspace(0.01, 0.3, 20)  # 1% to 30% churn
        x_label = "Monthly Churn Rate (%)"
        x_values = param_range * 100  # Convert to percentage for display
        
        ltv_values = []
        cac_values = []
        ratio_values = []
        
        for param in param_range:
            temp_results = calculate_economics(
                collection_size, user_count, avg_free_queries, avg_paid_queries, query_price, revenue_share,
                clusters_per_document, chunks_per_cluster, embedding_tokens_per_chunk, vector_dimensions,
                input_tokens_per_query, output_tokens_per_query, embedding_cost_per_m_tokens,
                vector_storage_cost, query_cost_per_m_dimensions, ai_input_cost, ai_output_cost, param
            )
            ltv_values.append(temp_results["LTV"])
            cac_values.append(temp_results["CAC"]) 
            ratio_values.append(temp_results["LTV/CAC Ratio"])
            
    elif sensitivity_param == "Average Free Queries":
        param_range = np.linspace(0.1, min(20.0, float(max_free_queries)), 20)
        x_label = "Average Free Queries Per User"
        x_values = param_range
        
        ltv_values = []
        cac_values = []
        ratio_values = []
        
        for param in param_range:
            temp_results = calculate_economics(
                collection_size, user_count, param, avg_paid_queries, query_price, revenue_share,
                clusters_per_document, chunks_per_cluster, embedding_tokens_per_chunk, vector_dimensions,
                input_tokens_per_query, output_tokens_per_query, embedding_cost_per_m_tokens,
                vector_storage_cost, query_cost_per_m_dimensions, ai_input_cost, ai_output_cost, monthly_churn_rate
            )
            ltv_values.append(temp_results["LTV"])
            cac_values.append(temp_results["CAC"])
            ratio_values.append(temp_results["LTV/CAC Ratio"])
            
    elif sensitivity_param == "Average Paid Queries":
        param_range = np.linspace(0.1, 20.0, 20)
        x_label = "Average Paid Queries Per User"
        x_values = param_range
        
        ltv_values = []
        cac_values = []
        ratio_values = []
        
        for param in param_range:
            temp_results = calculate_economics(
                collection_size, user_count, avg_free_queries, param, query_price, revenue_share,
                clusters_per_document, chunks_per_cluster, embedding_tokens_per_chunk, vector_dimensions,
                input_tokens_per_query, output_tokens_per_query, embedding_cost_per_m_tokens,
                vector_storage_cost, query_cost_per_m_dimensions, ai_input_cost, ai_output_cost, monthly_churn_rate
            )
            ltv_values.append(temp_results["LTV"])
            cac_values.append(temp_results["CAC"])
            ratio_values.append(temp_results["LTV/CAC Ratio"])
            
    else:  # Query Price
        param_range = np.linspace(0.01, 1.0, 20)
        x_label = "Query Price ($)"
        x_values = param_range
        
        ltv_values = []
        cac_values = []
        ratio_values = []
        
        for param in param_range:
            temp_results = calculate_economics(
                collection_size, user_count, avg_free_queries, avg_paid_queries, param, revenue_share,
                clusters_per_document, chunks_per_cluster, embedding_tokens_per_chunk, vector_dimensions,
                input_tokens_per_query, output_tokens_per_query, embedding_cost_per_m_tokens,
                vector_storage_cost, query_cost_per_m_dimensions, ai_input_cost, ai_output_cost, monthly_churn_rate
            )
            ltv_values.append(temp_results["LTV"])
            cac_values.append(temp_results["CAC"])
            ratio_values.append(temp_results["LTV/CAC Ratio"])
    
    # Create a figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot LTV and CAC
    ax1.plot(x_values, ltv_values, label="LTV", color="#66b3ff", linewidth=2)
    ax1.plot(x_values, cac_values, label="CAC", color="#ff9999", linewidth=2)
    
    # Mark the current value
    if sensitivity_param == "Monthly Churn Rate":
        current_x = monthly_churn_rate * 100
    elif sensitivity_param == "Average Free Queries":
        current_x = avg_free_queries
    elif sensitivity_param == "Average Paid Queries":
        current_x = avg_paid_queries
    else:
        current_x = query_price
        
    ax1.scatter([current_x], [results["LTV"]], color="#66b3ff", s=100, zorder=5)
    ax1.scatter([current_x], [results["CAC"]], color="#ff9999", s=100, zorder=5)
    
    ax1.set_xlabel(x_label)
    ax1.set_ylabel("Value ($)")
    ax1.set_title("LTV and CAC")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot LTV/CAC Ratio
    ax2.plot(x_values, ratio_values, color="#99ff99", linewidth=2)
    ax2.axhline(y=3, color='green', linestyle='--', alpha=0.5, label="Target (3.0)")
    ax2.axhline(y=1, color='red', linestyle='--', alpha=0.5, label="Breakeven (1.0)")
    
    # Mark the current value
    ax2.scatter([current_x], [results["LTV/CAC Ratio"]], color="#99ff99", s=100, zorder=5)
    
    ax2.set_xlabel(x_label)
    ax2.set_ylabel("LTV/CAC Ratio")
    ax2.set_title("LTV/CAC Ratio")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Customer Payback Period
    st.subheader("Customer Payback Period")
    
    # Calculate payback period in months
    monthly_profit_per_user = (avg_paid_queries * query_price * (1 - revenue_share / 100)) - ((avg_free_queries + avg_paid_queries) * results['Cost Per Query'])
    
    if monthly_profit_per_user > 0:
        payback_period = results['CAC'] / monthly_profit_per_user
        st.metric("Payback Period", f"{payback_period:.1f} months", 
                  help="Number of months required to recoup the cost of customer acquisition")
        
        # Visualization of payback period
        fig_payback, ax_payback = plt.subplots(figsize=(10, 5))
        
        # Generate months
        months = range(0, int(max(24, payback_period * 1.5)))
        cumulative_profit = [-results['CAC']] + [monthly_profit_per_user * m - results['CAC'] for m in months]
        
        # Plot cumulative profit
        ax_payback.plot([-1] + list(months), cumulative_profit, color="#66b3ff", linewidth=2)
        ax_payback.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax_payback.axvline(x=payback_period, color='green', linestyle='--', alpha=0.7)
        
        ax_payback.annotate(f'Payback: {payback_period:.1f} months', 
                     xy=(payback_period, 0),
                     xytext=(payback_period + 1, -results['CAC']/2),
                     arrowprops=dict(facecolor='green', shrink=0.05, alpha=0.7),
                     color='green')
        
        ax_payback.set_xlabel('Months')
        ax_payback.set_ylabel('Cumulative Profit per Customer ($)')
        ax_payback.set_title('Customer Payback Period')
        ax_payback.grid(True, alpha=0.3)
        
        st.pyplot(fig_payback)
        
        if payback_period > 12:
            st.warning("⚠️ Long payback period! Consider optimizing your pricing model or reducing CAC.")
        elif payback_period > 6:
            st.info("ℹ️ Moderate payback period. There might be room for optimization.")
        else:
            st.success("✅ Good payback period! Your business recovers acquisition costs quickly.")
    else:
        st.error("❌ Cannot calculate payback period: monthly profit per user is negative.")
        st.write("Adjust your parameters to achieve positive monthly profit per user.")
    
    # Key optimization insights
    st.subheader("Key Optimization Strategies")
    
    optimization_insights = []
    
    if results['LTV/CAC Ratio'] < 3:
        # Strategies to improve ratio
        if monthly_churn_rate > 0.05:
            optimization_insights.append("🔄 **Reduce churn rate**: Lowering churn from {:.1f}% to {:.1f}% would increase LTV by {:.0f}%.".format(
                monthly_churn_rate*100, monthly_churn_rate*50, 100))
        
        if avg_free_queries > 2:
            cac_reduction = (avg_free_queries - 1) / avg_free_queries * 100
            optimization_insights.append("🎁 **Optimize free tier**: Reducing avg free queries from {:.1f} to {:.1f} would reduce CAC by {:.0f}%.".format(
                avg_free_queries, avg_free_queries - 1, cac_reduction))
        
        if avg_paid_queries < 5:
            ltv_increase = (5 / avg_paid_queries - 1) * 100 if avg_paid_queries > 0 else 100
            optimization_insights.append("💰 **Increase paid usage**: Growing paid queries per user from {:.1f} to 5.0 would increase LTV by {:.0f}%.".format(
                avg_paid_queries, ltv_increase))
            
        if query_price < 0.15:
            optimization_insights.append("💲 **Consider price adjustments**: Increasing price from ${:.2f} to $0.15 could improve profitability if demand is inelastic.".format(query_price))
    
    if optimization_insights:
        for insight in optimization_insights:
            st.write(insight)
    else:
        st.write("✅ Your customer economics look strong! Continue monitoring these metrics as your business grows.")

# Add a downloads section at the bottom
st.markdown("---")
st.header("Downloads")

# Create a string representation of the cost breakdown
cost_breakdown = f"""
# Cost Structure for {collection_size:,} documents

## Content Structure
- Documents: {collection_size:,}
- Clusters per Document: {clusters_per_document} 
- Chunks per Cluster: {chunks_per_cluster}
- Total Clusters: {collection_size * clusters_per_document:,}
- Total Chunks: {collection_size * clusters_per_document * chunks_per_cluster:,}

## User Metrics
- Monthly Active Users: {user_count:,}
- Average Free Queries Per User: {avg_free_queries:.2f}
- Average Paid Queries Per User: {avg_paid_queries:.2f}
- Total Monthly Free Queries: {results['Total Free Queries']:,.0f}
- Total Monthly Paid Queries: {results['Total Paid Queries']:,.0f}
- Monthly Churn Rate: {monthly_churn_rate*100:.1f}%

## Revenue
- Query Price: ${query_price:.3f}
- Monthly Revenue: ${results['Total Revenue']:,.2f}

## Costs
- Content Preparation Cost: ${results['One-time Embedding Cost']:,.2f}
- Content Storage Cost: ${results['Monthly Storage Cost']:,.2f}
- Query Processing Cost: ${results['Total Query Cost']:,.2f}
- Total Monthly Cost: ${results['Total Monthly Cost']:,.2f}
- Monthly Amortized Embedding: ${results['Monthly Amortized Embedding']:,.2f}
- Total Costs (with Amortization): ${results['Total Monthly Cost with Amortization']:,.2f}

## Profitability
- Gross Profit (before Amortization): ${results['Profit']:,.2f}
- Profit Margin (before Amortization): {results['Profit Margin (%)']}%
- Profit with Amortization: ${results['Profit with Amortization']:,.2f}
- Profit Margin with Amortization: {results['Profit Margin with Amortization (%)']}%

## Break-Even Analysis
- Break-Even Monthly Paid Queries: ${results['Break-even Paid Queries']:,.0f}
- Break-Even Users: ${results['Break-even Users']:,.0f}

## Customer Economics
- Customer Acquisition Cost (CAC): ${results['CAC']:,.2f}
- Customer Lifetime Value (LTV): ${results['LTV']:,.2f}
- LTV/CAC Ratio: {results['LTV/CAC Ratio']:.2f}
"""

# Create a download button for the cost breakdown
st.download_button(
    label="Download Cost Breakdown",
    data=cost_breakdown,
    file_name="cost_breakdown.md",
    mime="text/markdown"
)

# Display fixed costs explanation
with st.expander("Fixed Costs Explanation"):
    st.write("""
    Fixed costs are one-time or recurring costs that don't directly scale with usage. Examples include:
    
    1. **Development costs** - Building and maintaining the document retrieval system
    2. **Integration costs** - Connecting to existing systems and data sources
    3. **Security and compliance** - Meeting industry regulations and security standards
    4. **Support staff** - Customer service and technical support personnel
    5. **Marketing and sales** - Acquiring new customers
    
    The simulator amortizes these fixed costs over a 12-month period to calculate the total cost structure and profitability.
    """)

st.write("Made with ❤️ by Ramus Analytics")
st.write("Pricing models and economic simulations are estimates. Actual results may vary.") 