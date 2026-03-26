def chat(self, spark=None, query_id=None):
        """Launches an interactive chat session with context awareness."""
        print("🔍 Gathering cluster context and DAG metrics...")
        
        features = None
        code_context = None
        
        # 1. Gather Context
        if spark and query_id:
            features = extract_features_from_system_tables(spark, query_id)
            # Fetch the actual code executed from the system table
            try:
                row = spark.sql(f"SELECT statement_text FROM system.query.history WHERE query_id = '{query_id}'").collect()[0]
                code_context = row['statement_text']
            except Exception:
                code_context = "Could not retrieve statement_text."
        else:
            features = extract_features_from_logs(self.log_dir)
            
        if not features:
            print("⚠️ No execution metrics found. The agent will chat without DAG context.")
            features = {"status": "No metrics available"}

        # 2. Initialize Chat
        print("💬 OptiSpark Agent is online. Type 'exit' to quit.\n")
        chat_session = self.engine.start_chat(features, code_context)
        
        # 3. The REPL Loop
        while True:
            try:
                user_input = input("\n👤 You: ")
                if user_input.lower() in ['exit', 'quit', 'q']:
                    print("👋 OptiSpark shutting down.")
                    break
                
                if not user_input.strip():
                    continue
                    
                print("🤖 OptiSpark is thinking...")
                response = chat_session.send_message(user_input)
                
                print(f"\n✨ OptiSpark:\n{response.text}")
                print("-" * 50)
                
            except KeyboardInterrupt:
                print("\n👋 OptiSpark shutting down.")
                break
            except Exception as e:
                print(f"\n❌ Error connecting to Reasoning Engine: {str(e)}")