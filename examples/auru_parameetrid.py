"""
Veeauru parameetrite arvutamine Pythonis
"""
import CoolProp.CoolProp as cp # dokumentatsioon: http://www.coolprop.org/index.html

t = 540 + 273.15 # K
p = 170 * 100000 # Pa

h = cp.PropsSI('Hmolar', 'T', t, 'P', p, 'WATER') # J mol^1
print('Vee entalpia tingimustel {} K ja {} bar: {} J/mol'.format(t, p, h)) # format funktsiooniga saame lisada muutujad sõnesse
