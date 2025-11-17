import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random

# Page config - MUST BE FIRST
st.set_page_config(page_title="AI Finance Analyzer", page_icon="💰", layout="wide")

# Custom CSS for beautiful UI
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.75rem;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Title Section
st.markdown("<h1 class='main-header'>💰 AI Finance Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Upload your transactions and get AI-powered financial insights in seconds!</p>", unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # AI Provider Selection
    ai_provider = st.radio(
        "Choose AI Provider:",
        ["Rule-Based (Free)", "OpenAI GPT"],
        help="Rule-Based doesn't need an API key!"
    )
    
    if ai_provider == "OpenAI GPT":
        api_key = st.text_input("OpenAI API Key", type="password", 
                                help="Get your key from platform.openai.com/api-keys")
    else:
        api_key = None
        st.info("✅ Using free rule-based AI - no API key needed!")
    
    st.markdown("---")
    
    # Create sample data button
    if st.button("🎲 Generate Sample Data"):
        st.session_state['use_sample'] = True
        st.success("✅ Sample data loaded!")

# Function to generate sample data
def generate_sample_data():
    categories_expense = ['Food', 'Transport', 'Shopping', 'Entertainment', 'Utilities', 'Healthcare']
    categories_income = ['Salary', 'Freelance', 'Investment']
    
    data = []
    start_date = datetime.now() - timedelta(days=30)
    
    # Generate expenses
    for i in range(40):
        date = start_date + timedelta(days=random.randint(0, 30))
        category = random.choice(categories_expense)
        amount = -round(random.uniform(10, 200), 2)
        descriptions = {
            'Food': ['Restaurant', 'Groceries', 'Coffee', 'Takeout'],
            'Transport': ['Uber', 'Gas', 'Parking', 'Metro'],
            'Shopping': ['Clothes', 'Electronics', 'Books', 'Home'],
            'Entertainment': ['Movie', 'Concert', 'Gaming', 'Subscription'],
            'Utilities': ['Electricity', 'Water', 'Internet', 'Phone'],
            'Healthcare': ['Pharmacy', 'Doctor', 'Gym', 'Insurance']
        }
        description = random.choice(descriptions[category])
        data.append([date.strftime('%Y-%m-%d'), category, amount, description])
    
    # Generate income
    for i in range(3):
        date = start_date + timedelta(days=random.randint(0, 30))
        category = random.choice(categories_income)
        amount = round(random.uniform(1000, 5000), 2)
        description = f"{category} payment"
        data.append([date.strftime('%Y-%m-%d'), category, amount, description])
    
    return pd.DataFrame(data, columns=['Date', 'Category', 'Amount', 'Description'])

# Rule-based AI analysis function
def generate_rule_based_insights(total_income, total_expenses, net_savings, savings_rate, category_spending):
    insights = []
    recommendations = []
    warnings = []
    
    # Insights
    if savings_rate > 20:
        insights.append("💚 Excellent savings rate! You're saving over 20% of your income.")
    elif savings_rate > 10:
        insights.append("👍 Good savings rate. You're on the right track!")
    else:
        insights.append("⚠️ Low savings rate. Consider reducing expenses.")
    
    top_category = category_spending.iloc[0]
    insights.append(f"📊 Your highest spending category is {top_category['Category']} at ${top_category['Amount']:.2f}")
    
    if len(category_spending) >= 3:
        top_3_total = category_spending.head(3)['Amount'].sum()
        top_3_pct = (top_3_total / total_expenses) * 100
        insights.append(f"💡 Your top 3 spending categories account for {top_3_pct:.1f}% of total expenses")
    
    # Recommendations
    if savings_rate < 20:
        recommendations.append(f"🎯 Try to reduce {top_category['Category']} spending by 15% to improve savings")
    
    food_spending = category_spending[category_spending['Category'] == 'Food']['Amount'].sum()
    if food_spending > total_expenses * 0.3:
        recommendations.append("🍽️ Food expenses are high. Consider meal planning and cooking at home more often")
    
    recommendations.append(f"💰 Set up automatic transfers of ${net_savings * 0.5:.2f} per month to a savings account")
    
    # Warnings
    if net_savings < 0:
        warnings.append("🚨 WARNING: You're spending more than you earn! Take immediate action to reduce expenses.")
    elif savings_rate < 5:
        warnings.append("⚠️ Very low savings rate. You may struggle with unexpected expenses.")
    
    # Format output
    output = "### 💡 Key Insights\n"
    for insight in insights:
        output += f"- {insight}\n"
    
    output += "\n### 🎯 Recommendations\n"
    for rec in recommendations:
        output += f"- {rec}\n"
    
    if warnings:
        output += "\n### ⚠️ Concerns\n"
        for warn in warnings:
            output += f"- {warn}\n"
    
    return output

# OpenAI API function (updated for v1.0+)
def generate_openai_insights(api_key, total_income, total_expenses, net_savings, savings_rate, category_spending, avg_transaction, transaction_count):
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        
        top_expenses = category_spending.head(3)
        
        prompt = f"""You are a professional financial advisor. Analyze this financial data and provide insights:

FINANCIAL SUMMARY:
- Total Income: ${total_income:,.2f}
- Total Expenses: ${total_expenses:,.2f}
- Net Savings: ${net_savings:,.2f}
- Savings Rate: {savings_rate:.1f}%
- Average Transaction: ${avg_transaction:.2f}

TOP SPENDING CATEGORIES:
{top_expenses.to_string(index=False)}

TRANSACTION COUNT: {transaction_count} transactions

Please provide:
1. Three key insights about spending patterns
2. Three specific, actionable recommendations to improve financial health
3. One warning or concern if applicable

Format your response with clear sections and bullet points."""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful financial advisor who provides clear, actionable advice."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        return f"Error: {str(e)}\n\nPlease check your API key or switch to Rule-Based AI."

# File Upload or Sample Data
uploaded_file = st.file_uploader("📤 Upload your transaction CSV", type=['csv'])

df = None

if 'use_sample' in st.session_state and st.session_state['use_sample']:
    df = generate_sample_data()
    st.info("📊 Using sample data - Upload your own CSV to analyze your finances!")
elif uploaded_file:
    df = pd.read_csv(uploaded_file)

# Main Analysis Section
if df is not None:
    
    # Data preprocessing
    df['Date'] = pd.to_datetime(df['Date'])
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
    df = df.dropna(subset=['Amount'])
    df = df.sort_values('Date')
    
    # Calculate key metrics
    total_income = df[df['Amount'] > 0]['Amount'].sum()
    total_expenses = abs(df[df['Amount'] < 0]['Amount'].sum())
    net_savings = total_income - total_expenses
    savings_rate = (net_savings / total_income * 100) if total_income > 0 else 0
    
    # Display Key Metrics
    st.markdown("## 📊 Financial Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("💵 Total Income", f"${total_income:,.2f}")
    with col2:
        st.metric("💸 Total Expenses", f"${total_expenses:,.2f}")
    with col3:
        st.metric("💰 Net Savings", f"${net_savings:,.2f}")
    with col4:
        st.metric("📈 Savings Rate", f"{savings_rate:.1f}%")
    
    st.markdown("---")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🥧 Spending by Category")
        expenses_df = df[df['Amount'] < 0].copy()
        expenses_df['Amount'] = abs(expenses_df['Amount'])
        category_spending = expenses_df.groupby('Category')['Amount'].sum().reset_index()
        category_spending = category_spending.sort_values('Amount', ascending=False)
        
        fig = px.pie(category_spending, values='Amount', names='Category',
                    color_discrete_sequence=px.colors.sequential.RdBu,
                    hole=0.4)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Top Spending Categories")
        top_5 = category_spending.head(5)
        fig = px.bar(top_5, x='Amount', y='Category', orientation='h',
                    color='Amount',
                    color_continuous_scale='Reds')
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Daily cash flow
    st.subheader("📈 Daily Cash Flow Over Time")
    daily_flow = df.groupby('Date')['Amount'].sum().reset_index()
    
    fig = go.Figure()
    
    colors = ['red' if x < 0 else 'green' for x in daily_flow['Amount']]
    
    fig.add_trace(go.Bar(
        x=daily_flow['Date'],
        y=daily_flow['Amount'],
        marker_color=colors,
        name='Daily Flow'
    ))
    
    fig.update_layout(
        title='Income (Green) vs Expenses (Red)',
        xaxis_title='Date',
        yaxis_title='Amount ($)',
        height=400,
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Recent Transactions Table
    st.markdown("---")
    st.subheader("📋 Recent Transactions")
    
    # Format the display
    display_df = df.sort_values('Date', ascending=False).head(10).copy()
    display_df['Amount'] = display_df['Amount'].apply(lambda x: f"${x:,.2f}")
    display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
    
    st.dataframe(display_df, use_container_width=True, height=300)
    
    # AI Analysis Section
    st.markdown("---")
    st.markdown("## 🤖 AI-Powered Insights")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.write(f"Get personalized financial advice powered by {ai_provider}")
    with col2:
        analyze_button = st.button("🚀 Generate AI Analysis", type="primary")
    
    if analyze_button:
        if ai_provider == "OpenAI GPT" and not api_key:
            st.error("⚠️ Please enter your OpenAI API key in the sidebar or switch to Rule-Based AI!")
        else:
            with st.spinner("🧠 AI is analyzing your financial data..."):
                avg_transaction = df['Amount'].mean()
                
                if ai_provider == "Rule-Based (Free)":
                    ai_response = generate_rule_based_insights(
                        total_income, total_expenses, net_savings, 
                        savings_rate, category_spending
                    )
                else:
                    ai_response = generate_openai_insights(
                        api_key, total_income, total_expenses, net_savings,
                        savings_rate, category_spending, avg_transaction, len(df)
                    )
                
                st.markdown("### 💡 AI Financial Analysis")
                st.success(ai_response)
                
                # Additional predictions
                st.markdown("### 🔮 Predictions")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    daily_expense_avg = total_expenses / 30
                    predicted_month = daily_expense_avg * 30
                    st.metric("Next Month Spending", f"${predicted_month:.2f}")
                
                with col2:
                    days_to_goal = int(5000 / net_savings * 30) if net_savings > 0 else 0
                    st.metric("Days to Save $5K", f"{days_to_goal} days" if days_to_goal > 0 else "Set goal")
                
                with col3:
                    yearly_savings = net_savings * 12
                    st.metric("Yearly Savings Projection", f"${yearly_savings:,.2f}")

    # Chatbot Section
    st.markdown("---")
    st.markdown("## 💬 Chat with Your Finance AI")
    
    user_question = st.text_input("Ask anything about your finances...", 
                                  placeholder="e.g., How can I save more on food?")
    
    if user_question:
        if ai_provider == "OpenAI GPT" and not api_key:
            st.warning("⚠️ Please add your OpenAI API key or switch to Rule-Based AI")
        else:
            with st.spinner("🤔 Thinking..."):
                if ai_provider == "Rule-Based (Free)":
                    # Simple rule-based responses
                    question_lower = user_question.lower()
                    
                    if 'food' in question_lower or 'eat' in question_lower:
                        response = f"Based on your data, you're spending ${category_spending[category_spending['Category'] == 'Food']['Amount'].sum():.2f} on food. Try meal planning, cooking at home, and limiting restaurant visits to save 20-30%."
                    elif 'save' in question_lower or 'saving' in question_lower:
                        response = f"Your current savings rate is {savings_rate:.1f}%. Aim for at least 20% by cutting your top expense category ({category_spending.iloc[0]['Category']}) by 15%."
                    elif 'transport' in question_lower or 'uber' in question_lower or 'gas' in question_lower:
                        transport_spending = category_spending[category_spending['Category'] == 'Transport']['Amount'].sum()
                        response = f"Transportation costs are ${transport_spending:.2f}. Consider carpooling, public transit, or biking to reduce these expenses."
                    else:
                        response = f"Your top spending area is {category_spending.iloc[0]['Category']} (${category_spending.iloc[0]['Amount']:.2f}). Focus on reducing this category first for maximum impact on your savings."
                    
                    st.info(f"🤖 **AI Response:** {response}")
                else:
                    try:
                        from openai import OpenAI
                        client = OpenAI(api_key=api_key)
                        
                        context = f"""Financial Data Summary:
- Income: ${total_income:.2f}
- Expenses: ${total_expenses:.2f}
- Savings: ${net_savings:.2f}
- Categories: {', '.join(category_spending['Category'].tolist())}"""

                        response = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                {"role": "system", "content": f"You are a financial advisor. Here's the user's data: {context}"},
                                {"role": "user", "content": user_question}
                            ],
                            max_tokens=200
                        )
                        
                        st.info(f"🤖 **AI Response:** {response.choices[0].message.content}")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

else:
    # Welcome screen
    st.info("👆 **Upload a CSV file or click 'Generate Sample Data' in the sidebar to get started!**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🎯 Features")
        st.markdown("""
        - Real-time financial analysis
        - AI-powered insights (Free!)
        - Beautiful visualizations
        - Spending predictions
        - Interactive chatbot
        """)
    
    with col2:
        st.markdown("### 🚀 Benefits")
        st.markdown("""
        - Track spending patterns
        - Get personalized advice
        - Predict future expenses
        - Improve savings rate
        - Make better decisions
        """)
    
    with col3:
        st.markdown("### 💡 Use Cases")
        st.markdown("""
        - Personal budgeting
        - Expense tracking
        - Financial planning
        - Savings goals
        - Investment decisions
        """)

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #666;'>Built with ❤️ for Genesis Hackathon | Powered by AI & Streamlit</p>", unsafe_allow_html=True)
