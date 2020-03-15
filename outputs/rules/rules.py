#This rule was reconstructed from outputs/rules/rules.json
def findDecision(obj):
	if obj[1]<=2:
		if obj[5] == '1':
			return 0
		elif obj[5] == '2':
			return 0
		elif obj[5] == '5':
			if obj[2]<=2:
				if obj[3]<=1:
					return 0
				elif obj[3]>1:
					return 0.5
			elif obj[2]>2:
				return 1
		elif obj[5] == '?':
			return 0
		elif obj[5] == '3':
			return 0
		elif obj[5] == '4':
			return 0
		elif obj[5] == '10':
			return 0.75
		elif obj[5] == '7':
			return 1
	elif obj[1]>2:
		if obj[5] == '10':
			if obj[6]>3:
				return 1
			elif obj[6]<=3:
				if obj[0]<=6:
					if obj[7]>2:
						return 0.6666666666666666
					elif obj[7]<=2:
						return 0.5
				elif obj[0]>6:
					return 1
		elif obj[5] == '1':
			if obj[2]>2:
				if obj[8]<=1:
					if obj[4]>2:
						if obj[3]>1:
							if obj[6]>3:
								if obj[0]>6:
									return 0.75
								elif obj[0]<=6:
									return 1
							elif obj[6]<=3:
								return 0.6666666666666666
						elif obj[3]<=1:
							return 0.3333333333333333
					elif obj[4]<=2:
						return 0
				elif obj[8]>1:
					return 1
			elif obj[2]<=2:
				return 0
		elif obj[5] == '5':
			if obj[0]>6:
				return 1
			elif obj[0]<=6:
				if obj[2]>2:
					if obj[3]>1:
						if obj[4]>2:
							if obj[6]>3:
								if obj[7]>2:
									if obj[8]<=1:
										return 1
		elif obj[5] == '8':
			if obj[7]>2:
				return 1
			elif obj[7]<=2:
				return 0.75
		elif obj[5] == '3':
			if obj[2]>2:
				return 1
			elif obj[2]<=2:
				return 0
		elif obj[5] == '2':
			if obj[4]>2:
				if obj[7]>2:
					return 1
				elif obj[7]<=2:
					return 0
			elif obj[4]<=2:
				return 0
		elif obj[5] == '7':
			if obj[8]<=1:
				return 1
			elif obj[8]>1:
				return 0.6666666666666666
		elif obj[5] == '4':
			return 1
		elif obj[5] == '9':
			return 1
		elif obj[5] == '?':
			return 0.3333333333333333
		elif obj[5] == '6':
			return 1
