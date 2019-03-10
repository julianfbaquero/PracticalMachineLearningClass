#!/usr/bin/python

import pandas as pd
from sklearn.externals import joblib
import sys
import os
import category_encoders as ce
import numpy as np

def predict_proba(datos):

    clf = joblib.load(os.path.dirname(__file__) + '/precio_clf.pkl')

    data_ = pd.DataFrame([datos], columns=['Year','Mileage','State','Make','Model'])
    Big_data=data_
    print('Big data es:','\n',Big_data)
    
    # Create features
    #Big_data['Year_s']=data_[str('Year')]

    data1=Big_data['Year']
    total_year=np.arange(1997,2019)
    X_1=pd.get_dummies(total_year, prefix='Y')
    j1=0
    for i in total_year:
        if i==data1.iloc[0]:
            X1=X_1.iloc[j1,:]
            break
        j1+=1
    X1_df=pd.DataFrame(np.array(X1))
    X_1_df=X1_df.T
    print('x1 es:','\n',X_1_df)

    data2=pd.DataFrame(Big_data['State'])
    #print('state','\n',np.array(data2.iloc[0]))
    total_state=['MD', 'KY', 'SC', 'OK', 'TN', 'FL', 'NH', 'WI', 'NY', 'TX', 'NJ', 'MI', 'AL', 'CA', 'NC', 'GA', 'OR', 'OH', 'AR', 'VA', 'WA', 'IL', 'AZ', 'MA', 'CO', 'MN', 'KS', 'PA', 'MO', 'SD', 'IN', 'NE', 'UT', 'NM', 'HI', 'NV', 'DE', 'MS', 'ID', 'IA', 'ME', 'CT', 'MT', 'VT', 'WV', 'LA', 'ND', 'AK', 'RI', 'WY', 'DC']
    X_2 = ce.BinaryEncoder().fit_transform(total_state) 
    j2=0
    for i in total_state:
        if i==np.array(data2.iloc[0]):
            X2=X_2.iloc[j2,:]
            break
        j2+=1
    X2_df=pd.DataFrame(np.array(X2))
    X_2_df=X2_df.T
    print('x2 es:','\n',X_2_df)

    data3=pd.DataFrame(Big_data['Make'])
    total_make=['Nissan', 'Chevrolet', 'Hyundai', 'Jeep', 'Ford', 'Kia', 'Mercedes-Benz', 'Dodge', 'GMC', 'Toyota', 'Honda', 'Volkswagen', 'Cadillac', 'Volvo', 'BMW', 'Subaru', 'Chrysler', 'Buick', 'Ram', 'Lexus', 'Porsche', 'Audi', 'Lincoln', 'MINI', 'INFINITI', 'Scion', 'Land', 'Acura', 'Mazda', 'Mercury', 'Mitsubishi', 'Pontiac', 'Jaguar', 'Bentley', 'Suzuki', 'FIAT', 'Tesla', 'Freightliner', 'Saturn']
    X_3 = ce.BinaryEncoder().fit_transform(total_make)
    j3=0
    for i in total_make:
        if i==np.array(data3.iloc[0]):
            X3=X_3.iloc[j3,:]
            break
        j3+=1
    X3_df=pd.DataFrame(np.array(X3))
    X_3_df=X3_df.T
    print('x3 es:','\n',X_3_df)

    data4=pd.DataFrame(Big_data['Model'])
    total_model=['MuranoAWD', 'CamaroCoupe', 'Santa', 'Grand', 'Wrangler', 'F-1504WD', 'ExplorerXLT', 'Sonata4dr', 'SorentoSX', 'M-ClassML350', 'JourneyFWD', 'Super', 'Sierra', 'Silverado', 'CamryLE', 'OdysseyEX-L', 'Pathfinder4WD', 'AcadiaFWD', 'EscapeSE', 'Express', 'F-150XLT', 'EdgeSEL', 'Passat4dr', 'EdgeSport', 'FlexLimited', 'CTS', 'S60T5', 'CompassSport', '5', 'Impreza', 'RogueFWD', 'Civic', 'CruzeSedan', 'Accord', 'AccordLX', 'PriusTwo', 'SorentoEX', 'PilotEX-L', 'Town', 'Legacy', 'RegalTurbo', 'Yukon2WD', '25004WD', 'IS', 'Tacoma2WD', 'X3xDrive28i', 'Ranger2WD', 'Focus4dr', 'Escape4WD', 'SonataLimited', '200Limited', 'CR-VLX', 'CayenneAWD', 'MalibuLT', 'CherokeeLimited', 'TundraSR5', 'F-150Lariat', 'Impala4dr', 'Q7quattro', 'Tiguan2WD', 'CR-VEX', 'Suburban4WD', 'F-350XLT', 'Regal4dr', '350Z2dr', 'XC60T6', 'ElantraLimited', '300Touring', 'Camry4dr', 'Liberty4WD', 'MustangPremium', 'Legacy2.5i', 'RX', 'FusionSE', 'F-150STX', 'Armada4WD', 'Ram', 'Jetta', 'Yukon', 'Altima4dr', 'CivicLX', 'GS', 'Tacoma4WD', 'CorollaS', 'ExplorerLimited', 'DurangoAWD', 'Navigator4dr', 'CorollaLE', 'TT2dr', 'SedonaLX', 'CivicEX-L', 'EscapeFWD', 'Sentra4dr', 'LX', '3004dr', 'EnclaveLeather', 'Tahoe4dr', '300300S', 'Maxima4dr', 'Legacy3.6R', 'EdgeLimited', 'CamrySE', '3', 'AcadiaAWD', 'Malibu1LT', 'Sorento2WD', 'F-1502WD', 'E-ClassE', 'Cooper', 'Soul+', 'RAV4FWD', 'CamaroConvertible', 'AvalonLimited', 'OptimaLX', 'G37', 'Tundra', 'ExpeditionXLT', 'FocusSE', 'F-150FX4', 'EquinoxAWD', 'PatriotSport', 'EquinoxFWD', 'SequoiaPlatinum', 'Tahoe2WD', 'TaurusSEL', 'Explorer4WD', 'tC2dr', 'ChargerSXT', 'ExpeditionLimited', '15004WD', 'Elantra', 'Corvette2dr', 'CompassLatitude', 'Suburban2WD', 'SiennaSE', 'MalibuLS', 'PatriotLatitude', 'CamryXLE', 'TundraLimited', 'TerrainAWD', 'Elantra4dr', 'Durango2WD', 'Rover', 'MustangGT', 'FJ', 'Malibu', 'XC90AWD', '6', 'EnclavePremium', 'Challenger2dr', 'ES', 'TL4dr', '4Runner4WD', 'F-250XLT', 'Dakota2WD', 'Expedition4WD', 'ISIS', 'AccordEX-L', 'CTS4dr', '200LX', 'Mustang2dr', 'WRXPremium', 'CX-9Grand', 'ForteLX', 'Frontier4WD', 'CanyonCrew', 'Outback2.5i', 'Titan2WD', 'Transit', 'Forester2.5X', 'CX-7FWD', 'LibertySport', 'CorollaL', 'Prius5dr', 'Milan4dr', 'OdysseyEX', 'EscaladeAWD', 'CTS-V', 'PacificaTouring', 'A64dr', 'E-ClassE350', '35004WD', 'Sienna5dr', 'DTS4dr', 'A44dr', 'Q5quattro', 'RAV44WD', 'LS', 'CR-VSE', 'TacomaBase', 'Camaro2dr', 'LaCrosse4dr', 'SportageLX', 'SRXLuxury', 'Titan', 'C-Class4dr', 'Passat', 'C-ClassC', 'PriusThree', 'GXGX', 'SonicSedan', 'Yukon4WD', 'Mazda64dr', 'CR-V4WD', 'C-ClassC300', 'Camry', 'LXLX', '300Limited', 'TraverseFWD', 'FiestaSE', 'CamryL', 'SonicHatch', 'ImpalaLT', 'CorvetteCoupe', 'Escalade', 'Excursion137"', 'Murano2WD', '1500Tradesman', 'RidgelineRTL', 'WranglerSahara', '4RunnerSR5', 'OptimaEX', 'Highlander4WD', 'TucsonAWD', 'CivicEX', 'FX35AWD', 'SorentoLX', 'PilotLX', 'Mazda35dr', 'LSLS', 'SonataSE', '15002WD', 'Eclipse3dr', 'JourneyAWD', 'Outlander', 'F-150XL', 'ImprezaSport', 'FlexSEL', 'Tahoe4WD', 'AccordLX-S', 'FusionS', 'CR-VEX-L', 'SiennaXLE', 'Corolla4dr', 'Prius', 'MDXAWD', 'Malibu4dr', 'Tundra2WD', 'X5AWD', 'Pilot4WD', 'X3AWD', 'WranglerSport', 'TLAutomatic', 'Impreza2.0i', 'Sprinter', 'Wrangler2dr', 'OptimaSX', 'MDX4WD', 'HighlanderFWD', '1500Laramie', 'SportageAWD', 'X5xDrive35i', 'SL-ClassSL500', 'TerrainFWD', 'EscapeS', 'SoulBase', 'RXRX', 'FusionHybrid', 'RegalPremium', 'Expedition', 'F-150Platinum', 'Forester4dr', 'Colorado2WD', 'TahoeLT', 'RDXAWD', 'Fusion4dr', 'SequoiaSR5', 'HighlanderLimited', 'TSXAutomatic', 'Frontier2WD', 'SiennaLimited', 'JourneySXT', 'PriusFour', 'TucsonLimited', '1', 'CruzeLT', 'Durango4dr', 'GX', 'OdysseyTouring', 'Navigator', 'SC', 'F-150SuperCrew', 'ExplorerBase', 'Xterra4WD', 'ColoradoCrew', 'RioLX', 'Optima4dr', 'Charger4dr', 'RAV4LE', 'Savana', 'TucsonFWD', 'MustangDeluxe', 'Escalade2WD', 'MustangBase', 'GSGS', 'Colorado4WD', 'Titan4WD', '300Base', 'Wrangler4WD', 'GTI4dr', 'Navigator2WD', 'QX564WD', 'Pilot2WD', 'TundraBase', 'TaurusLimited', '200S', 'Econoline', 'LaCrosseFWD', 'WranglerX', 'SportageEX', 'FocusTitanium', 'ESES', 'Mazda34dr', 'RDXFWD', 'Eos2dr', 'RidgelineSport', 'New', 'Compass4WD', 'ChallengerR/T', 'Lucerne4dr', 'AvalonXLE', 'YarisBase', 'Taurus4dr', 'PriusOne', 'MX5', 'EnclaveConvenience', 'F-250XL', 'Avalon4dr', 'CR-V2WD', 'Golf', 'Navigator4WD', 'TiguanS', 'Avalanche4WD', '9112dr', '4Runner4dr', 'Escalade4dr', '7', 'TacomaPreRunner', 'Versa4dr', '300300C', 'C702dr', 'AccordEX', 'TiguanSEL', 'Touareg4dr', 'ImpalaLS', 'MKXAWD', 'FocusSEL', 'MustangShelby', 'PilotEX', 'ForteEX', 'PilotTouring', 'DurangoSXT', 'RAV4XLE', '4RunnerLimited', 'CherokeeSport', 'S2000Manual', 'F-150FX2', 'TraverseAWD', 'RX-84dr', 'PathfinderS', 'xB5dr', 'OdysseyLX', '4RunnerRWD', 'A34dr', 'XC60AWD', 'MKZ4dr', 'WranglerRubicon', 'Monte', 'RangerSuperCab', 'SequoiaLimited', 'ChargerSE', 'Sequoia4dr', 'Forte', 'CC4dr', 'GTI2dr', 'EdgeSE', 'Yaris', 'C-ClassC350', 'Matrix5dr', 'G35', '4RunnerTrail', 'S804dr', 'G64dr', '911', 'Highlander', 'Boxster2dr', 'Element4WD', 'CTCT', 'RAV4Sport', 'EscapeXLT', 'ForteSX', 'Focus5dr', 'F-250Lariat', 'ExplorerFWD', 'HighlanderBase', 'CX-9FWD', 'Dakota4WD', 'FocusS', 'Yukon4dr', 'Outback3.6R', 'Quest4dr', 'MuranoS', 'X1xDrive28i', 'Ranger4WD', 'RAV4Base', 'RAV4', 'CX-9AWD', 'Highlander4dr', 'CivicSi', 'XJ4dr', 'F-250King', 'CT', 'Sportage2WD', 'WRXSTI', 'AccordSE', 'FlexSE', 'A8', 'S44dr', 'Continental', 'AvalonTouring', 'TiguanSE', 'F-350Lariat', 'Escape4dr', 'QX562WD', 'Outlander4WD', 'Caliber4dr', 'Armada2WD', 'Element2WD', 'CamryBase', '4Runner2WD', 'Tundra4WD', 'RAV4Limited', 'Suburban4dr', 'Frontier', 'Genesis', 'F-350XL', 'SedonaEX', 'Accent4dr', 'FusionSEL', 'Vibe4dr', 'PilotSE', 'MKXFWD', 'XC704dr', 'WRXBase', 'PacificaLimited', 'SiennaLE', 'CR-ZEX', 'LaCrosseAWD', 'PatriotLimited', 'ColoradoExtended', 'Explorer', 'SportageSX', 'FitSport', '500Pop', 'RAV44dr', 'Galant4dr', 'F-150Limited', 'PT', 'Sequoia4WD', 'Expedition2WD', 'Pathfinder2WD', 'E-ClassE320', 'Land', 'GLI4dr', 'PathfinderSE', 'Xterra4dr', 'CX-9Touring', 'RegalGS', 'CorvetteConvertible', 'F-150King', 'Sedona4dr', 'EscapeLImited', 'TaurusSE', 'Avalanche2WD', 'Versa5dr', 'YarisLE', 'Cayman2dr', 'Cobalt4dr', 'Canyon4WD', 'XF4dr', 'PriusBase', 'Explorer4dr', 'VeracruzFWD', 'xD5dr', 'Lancer4dr', 'XC60FWD', 'TaurusSHO', 'CompassLimited', '200Touring', 'STS4dr', 'TahoeLS', 'CanyonExtended', 'Azera4dr', 'Cobalt2dr', 'PriusFive', 'Model', 'WRXLimited', 'FocusST', 'Canyon2WD', 'HighlanderSE', 'Patriot4WD', 'Outlander2WD', 'VeracruzAWD', 'F-350King', 'SLK-ClassSLK350', 'EscapeLimited', '25002WD', 'Xterra2WD', 'ExplorerEddie', 'XC90FWD', 'Yaris4dr', 'LibertyLimited', 'XK2dr', 'XC90T6', 'FiestaS']
    X_4 = ce.BinaryEncoder().fit_transform(total_model)
    j4=0
    for i in total_model:
        if i==np.array(data4.iloc[0]):
            X4=X_4.iloc[j4,:]
            break
        j4+=1
    X4_df=pd.DataFrame(np.array(X4))
    X_4_df=X4_df.T
    print('x4 es:','\n',X_4_df)
    
    New_data =pd.concat([Big_data[['Mileage']],X_1_df,X_2_df,X_3_df,X_4_df],axis=1)
    print(New_data)
    # Make prediction
    #p1 = clf.predict_proba(New_data.drop('url', axis=1))[0,1]
    p1 = clf.predict(New_data)

    return p1


if __name__ == "__main__":

    if len(sys.argv) == 1:
        print('Por favor ingrese los datos')

    else:

        url = sys.argv[1]

        p1 = predict_proba(datos)

        print(datos)
        print('El valor es: ', p1)
