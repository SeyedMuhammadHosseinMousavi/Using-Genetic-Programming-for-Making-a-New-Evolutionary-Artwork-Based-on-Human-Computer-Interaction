function emp=AssimilateColonies(emp)

    global ProblemSettings;
    CostFunction=ProblemSettings.CostFunction;
    VarSize=ProblemSettings.VarSize;
    VarMin=ProblemSettings.VarMin;
    VarMax=ProblemSettings.VarMax;
    
    global ICASettings;
    beta=ICASettings.beta;
    
    nEmp=numel(emp);
    for k=1:nEmp
        for i=1:emp(k).nCol
            
            NewPos = emp(k).Col(i).Position + beta*rand(VarSize).*(emp(k).Imp.Position-emp(k).Col(i).Position);
			NewPos = max(NewPos,VarMin);
			NewPos = min(NewPos,VarMax);
            
			emp(k).Col(i).Position = NewPos;
			
            emp(k).Col(i).Cost = CostFunction(emp(k).Col(i).Position);
            
        end
    end

end
